import os
import torch
import transformers
from transformers import AutoTokenizer
from transformers import TrainingArguments

from llmtools.llms.autollm import AutoLLMForCausalLM
from llmtools.engine.lora.config import FinetuneConfig
from llmtools.data import TrainSAD
from llmtools.engine.hf.trainer import Trainer
from llmtools.engine.lora.peft import quant_peft

from accelerate import Accelerator

os.environ['NUMEXPR_MAX_THREADS'] = '24'

# model config
model_name = 'relaxml/Llama-1-30b-E8P-2Bit' # HF dir.

# device_map = "auto"
device_index = Accelerator().process_index
device_map = {"": device_index}

# load model
llm, quip_config = AutoLLMForCausalLM.from_pretrained(model_name, load_in_quip=True, device_map=device_map)
llm.eval()

#* AutoTokenizer is the lateste version of tokenizer, avoid tokenizer warning and error *#
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device_map, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

llm.eval()

# finetune training config
mbatch_size_per_device=1
batch_size= 16 
epochs=3
lr=1e-3
cutoff_len=256
lora_r=8
lora_alpha=16
lora_dropout=0.05
val_set_size=0.2
warmup_steps=0
save_steps=10
save_total_limit=3
logging_steps=1

data_type = 'alpaca'
dataset = None # will load alpaca from HF
adapter_path = './llama1-30b-alpaca-ddp'

# set up finetuning config
tune_config = FinetuneConfig(
    dataset=dataset, 
    ds_type=data_type, 
    lora_out_dir=adapter_path, 
    mbatch_size=mbatch_size_per_device,
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
    task_type="CAUSAL_LM",
    r=tune_config.lora_r,
    lora_alpha=tune_config.lora_alpha,
    lora_dropout=tune_config.lora_dropout,
    bias="none",
    target_modules=["qkv_proj"],
)

# Data Parallel Training
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    num_of_gpus = torch.cuda.device_count()
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = tune_config.batch_size // (tune_config.mbatch_size*num_of_gpus)
else:
    gradient_accumulation_steps = tune_config.batch_size 

print("gradient_accumulation_steps: ", gradient_accumulation_steps)

# create a new lora from config
model = quant_peft.get_peft_model(llm, lora_config)

print(model)

# load stanford alpaca data
data = TrainSAD(
    tune_config.dataset, 
    tune_config.val_set_size, 
    tokenizer, 
    tune_config.cutoff_len
)
data.prepare_data() # this tokenizes the dataset

# training args
training_arguments = TrainingArguments(
    per_device_train_batch_size=tune_config.mbatch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=tune_config.warmup_steps,
    num_train_epochs=tune_config.epochs,
    learning_rate=tune_config.lr,
    fp16=True,
    logging_steps=tune_config.logging_steps,
    evaluation_strategy="no",
    save_strategy="steps",
    eval_steps=None, #None
    save_steps=tune_config.save_steps,
    output_dir=tune_config.lora_out_dir,
    save_total_limit=tune_config.save_total_limit,
    load_best_model_at_end=False,
    ddp_find_unused_parameters=False if tune_config.ddp else None,
)

# start trainer
trainer = Trainer(
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


# start training
checkpoint_dir = tune_config.lora_out_dir
if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

# Save Model
model.save_pretrained(tune_config.lora_out_dir)

