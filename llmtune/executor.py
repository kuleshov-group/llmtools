import os
import torch
import torch.nn as nn

from llmtune.config import DEV, LLAMA_MODELS, OPT_MODELS, get_llm_config
from llmtune.llms.llama.model import load_llama
from llmtune.llms.opt.model import load_opt
from llmtune.engine.data import TrainTxt, TrainSAD, TrainGPT4All
from llmtune.engine.data.calibration import get_calibration_loaders
from llmtune.engine.lora.peft import quant_peft
from llmtune.engine.quant.algorithm import executor as quant_executor
from llmtune.utils import to_half_precision

def load_llm(model, weights, groupsize=-1):
    llm_config = get_llm_config(model)
    if model in LLAMA_MODELS:
        llm, tokenizer = load_llama(llm_config, weights, groupsize)
    elif model in OPT_MODELS:
        llm, tokenizer = load_opt(llm_config, weights)
    else:
        raise ValueError(f"Invalid model name: {model}")
    llm.eval()
    return llm, tokenizer

def load_adapter(llm, adapter_path=None, lora_config=None):
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

def load_data(config, tokenizer):
    if config.ds_type == "alpaca":
        data = TrainSAD(
            config.dataset, config.val_set_size, tokenizer, config.cutoff_len
        )
    elif config.ds_type == "gpt4all":
        raise NotImplementedError('GPT4All dataset currently not supported')
        data = TrainGPT4All(
            config.dataset, config.val_set_size, tokenizer, config.cutoff_len
        )
    else:
        raise ValueError(f"Invalid data name: {config.ds_type}")
    # data.prepare_data(
    #     thd=config.txt_row_thd, use_eos_token=config.use_eos_token
    # )
    data.prepare_data()
    return data

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

    data = load_data(tune_config, tokenizer)

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

def quantize(
    llm_config, dataset, nsamples, wbits, groupsize, percdamp, seed, weights
):
    model = load_llama_unquantized(llm_config)
    model.eval()
    dataloader, _ = get_calibration_loaders(
        dataset, 
        nsamples=nsamples, 
        seed=seed, 
        model=model, 
        seqlen=model.seqlen
    )

    tick = time.time()
    quantizers = quant_executor.quantize_llama(model, dataloader, DEV)
    print(f'Quantization time (s): {time.time() - tick}')

    quant_executor.pack_llama(model, quantizers, wbits)
    torch.save(model.state_dict(), weights) 
    print(f'Model weights saved to: {weights}')
