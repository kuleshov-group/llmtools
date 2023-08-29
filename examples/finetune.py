import os
import torch
import transformers
from transformers import AutoTokenizer
from llmtune.llms.autollm import AutoLLMForCausalLM
from llmtune.engine.lora.config import FinetuneConfig
from llmtune.data import TrainSAD
from llmtune.engine.lora.peft import quant_peft
from llmtune.utils import to_half_precision

# model config
model_name = './llama-7b-quantized' # can generate this via quantize.py
tokenizer_name = 'huggyllama/llama-13b'
DEV = 'cuda'

# load model
transformers.logging.set_verbosity_info()
llm = AutoLLMForCausalLM.from_pretrained(model_name)
llm.eval()
llm = llm.to(DEV)
llm = to_half_precision(llm)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token_id = 0

# finetune training config
mbatch_size=1
batch_size=2
epochs=3
lr=2e-4
cutoff_len=256
lora_r=8
lora_alpha=16
lora_dropout=0.05
val_set_size=0.2
warmup_steps=50
save_steps=50
save_total_limit=3
logging_steps=10

data_type = 'alpaca'
dataset = None # will load alpaca from HF
adapter_path = './llama-7b-quantized-lora'

# set up finetuning config
tune_config = FinetuneConfig(
    dataset=dataset, 
    ds_type=data_type, 
    lora_out_dir=adapter_path, 
    mbatch_size=mbatch_size,
    batch_size=batch_size,
    epochs=epochs, 
    lr=lr,
    cutoff_len=cutoff_len,
    lora_r=lora_r, 
    lora_alpha=lora_alpha, 
    lora_dropout=lora_dropout,
    val_set_size=val_set_size,
    warmup_steps=warmup_steps,
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    logging_steps=logging_steps,
)

# set up lora config    
lora_config = quant_peft.LoraConfig(
    r=tune_config.lora_r,
    lora_alpha=tune_config.lora_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=tune_config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

# create a new lora from config
model = quant_peft.get_peft_model(llm, lora_config)

# load stanford alpaca data
data = TrainSAD(
    tune_config.dataset, 
    tune_config.val_set_size, 
    tokenizer, 
    tune_config.cutoff_len
)
data.prepare_data() # this tokenizes the dataset

# training args
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

# start trainer
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

