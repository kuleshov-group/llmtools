import os
import time
import torch

from llmtools.config import DEV
from llmtools.utils import to_half_precision

def load_llm(model_name_or_path):
    from llmtools.llms.autollm import AutoLLMForCausalLM
    llm = AutoLLMForCausalLM.from_pretrained(model_name_or_path)
    return llm

def load_tokenizer(model_name_or_path, llm_config=None):
    from llmtools.llms.autollm import get_default_tokenizer
    if llm_config is not None:
        model_type = llm_config.model_type
    else:
        model_type = None
    return get_default_tokenizer(model_name_or_path, model_type)

def load_adapter(llm, adapter_path=None, lora_config=None):
    from llmtools.engine.lora.peft import quant_peft
    if adapter_path is None and lora_config is not None:
        model = quant_peft.get_peft_model(llm, lora_config)
    elif adapter_path is not None and lora_config is None:
        model = quant_peft.PeftModel.from_pretrained(
            llm, adapter_path, 
            device_map='auto',
            torch_dtype=torch.float32
        )
        print(adapter_path, 'loaded')
    else:
        ValueError('Need to specify adapter_path or lora_config')
    return model

def generate(
    llm, tokenizer, prompt, min_length, max_length, temperature, top_k, top_p
):
    llm.to(DEV)
    llm = to_half_precision(llm)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEV)

    with torch.no_grad():
        generated_ids = llm.generate(
            inputs=input_ids,
            do_sample=True,
            min_length=min_length,
            max_length=max_length,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
    return tokenizer.decode([el.item() for el in generated_ids[0]])    

def finetune(llm, tokenizer, tune_config):
    import transformers
    from llmtools.data import load_finetuning_data
    from llmtools.engine.lora.peft import quant_peft
    transformers.logging.set_verbosity_info()
    tokenizer.pad_token_id = 0
    
    lora_config = quant_peft.LoraConfig(
        r=tune_config.lora_r,
        lora_alpha=tune_config.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=tune_config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = load_adapter(llm, lora_config=lora_config)
    model.print_trainable_parameters()

    data = load_finetuning_data(tune_config, tokenizer)

    training_arguments = transformers.TrainingArguments(
        per_device_train_batch_size=tune_config.mbatch_size,
        gradient_accumulation_steps=tune_config.gradient_accumulation_steps,
        warmup_steps=tune_config.warmup_steps,
        num_train_epochs=tune_config.epochs,
        learning_rate=tune_config.lr,
        fp16=True,
        logging_steps=tune_config.logging_steps,
        evaluation_strategy="no",
        save_strategy="steps",
        eval_steps=None,
        save_steps=tune_config.save_steps,
        output_dir=tune_config.lora_out_dir,
        save_total_limit=tune_config.save_total_limit,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False if tune_config.ddp else None,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data.train_data,
        eval_dataset=data.val_data,
        args=training_arguments,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )
    print(training_arguments.parallel_mode)
    model.config.use_cache = False

    # use half precision
    model = to_half_precision(model)

    # start training
    checkpoint_dir = tune_config.lora_out_dir
    if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save Model
    model.save_pretrained(tune_config.lora_out_dir)

def quantize(llm, config):
    from llmtools.data.calibration import get_calibration_loaders
    from llmtools.engine.quant.gptq.executor import GPTQAlgorithm

    llm.eval()
    dataloader, _ = get_calibration_loaders(
        config.dataset, 
        nsamples=config.nsamples, 
        seed=config.seed, 
        model=llm.base_model.name_or_path, 
        seqlen=llm.base_model.seqlen
    )

    gptq = GPTQAlgorithm(config)
    llm = gptq.quantize(llm, dataloader)

    llm.save_pretrained(config.save)
    print(f'Model weights saved to: {config.save}')

    