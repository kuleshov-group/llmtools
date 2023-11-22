import argparse
# Set up the argument parser
parser = argparse.ArgumentParser(description='Python script to work with models')
parser.add_argument('--model_name', type=str, help='Name of the model', required=True)
parser.add_argument('--adapter', type=str, help='Path to store adapter weight', required=True)
parser.add_argument('--mbatch_size', type=int, help='mbatch size for training', required=True)
parser.add_argument('--seed', type=int, help='model seed number', required=True)

# Parse the arguments
args = parser.parse_args()

# Use the command line arguments in your script
print('Model Name:', args.model_name)
print('Adapter Path: ', args.adapter)
print('Seed: ', args.seed)
print('mbatch_size: ', args.mbatch_size)


import os
import torch
import transformers
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, DataCollatorForTokenClassification, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments, logging, TrainerCallback, TrainerState, TrainerControl, BitsAndBytesConfig
from llmtune.llms.autollm import AutoLLMForCausalLM
from llmtune.engine.lora.config import FinetuneConfig
from llmtune.engine.lora.peft import quant_peft
from llmtune.utils import to_half_precision
from datasets import load_dataset

from utils import *
from data import *


# os env setting
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# model config
model_name = args.model_name
tokenizer_name = 'facebook/opt-6.7b'
DEV = 'cuda'

transformers.logging.set_verbosity_info()


#* load model (QUIP) *#
if "quip" in model_name:
    llm, tokenizer, quip_config = AutoLLMForCausalLM.from_pretrained(model_name)
    llm.eval()

#* LoRA Config Set-UP*#

# finetune training config
MICRO_BATCH_SIZE=args.mbatch_size
BATCH_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 3  
LEARNING_RATE = 1e-3  # the Karpathy constant
CUTOFF_LEN = 128  # 128 accounts for about 95% of the data
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
VAL_SET_SIZE= 2000

# data/gpu config
seed = args.seed
set_random_seed(seed)
train_sample_rate = 1.0
val_sample_rate = 1.0

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1

# if ddp:
#     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
#     gradient_accumulation_steps = gradient_accumulation_steps // world_size

# adapter_path = '/share/kuleshov/vk379/alpacas/opt-7b-quantized-lora'
lora_out_dir = args.adapter

# set up lora config    
lora_config = quant_peft.LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

# ? Multi-gpu enabling (required testing) ? #
if not ddp and torch.cuda.device_count() > 1:
    llm.is_parallelizable = True
    llm.model_parallel = True


# create a new lora from config
breakpoint()
model = quant_peft.get_peft_model(llm, lora_config)

if not ddp and torch.cuda.device_count() > 1:
    print("GPU parallel acctivated")
    model.is_parallelizable = True
    model.model_parallel = True

# load stanford alpaca data
dataset = load_dataset('samsum')
train_records = dataset['train']
val_records = dataset['test']

## Config for SAMSum Dataset ##
model_type = "causal"
templates_path = "llama_lora_samsum.json"
only_target_loss = False
mode = "instruct"



#* Dataset Set-UP *#

if mode == "instruct":
    max_source_tokens_count = 205 # Changed depending on the dataset
    max_target_tokens_count = 45
    target_field = "summary"
    source_field = "" #does not matter. (original alpaca-lora paper has additional "input" alongside instruction: instruction-input-output vs. instruction-response)

    train_dataset = InstructDataset(
        train_records,
        tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        max_target_tokens_count=max_target_tokens_count,
        sample_rate=train_sample_rate,
        input_type=model_type,
        templates_path=templates_path,
        target_field=target_field,
        source_field=source_field,
        only_target_loss=only_target_loss
    )

    val_dataset = InstructDataset(
        val_records,
        tokenizer,
        max_source_tokens_count=max_source_tokens_count,
        max_target_tokens_count=max_target_tokens_count,
        sample_rate=val_sample_rate,
        input_type=model_type,
        templates_path=templates_path,
        target_field=target_field,
        source_field=source_field,
        only_target_loss=only_target_loss
    )

    ## Save the model
    dataloader_train = torch.utils.data.DataLoader(train_dataset)
    # torch.save(dataloader_train,'dataloader_train.pth')

    dataloader_val = torch.utils.data.DataLoader(val_dataset)
    # torch.save(dataloader_val,'dataloader_val.pth')

else:
    assert False

if "seq2seq" in model_type:
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
else:
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

print("INPUT_IDS")
print(data_collator([train_dataset[0], train_dataset[1]])["input_ids"][0])
print("MASK")
print(data_collator([train_dataset[0], train_dataset[1]])["attention_mask"][0])
print("LABELS")
print(data_collator([train_dataset[0], train_dataset[1]])["labels"][0])



# Model configs
model.config.num_beams = 5
if mode == "instruct":
    max_tokens_count = max_target_tokens_count + max_source_tokens_count + 1
model.config.max_length = max_tokens_count if model_type == "causal" else max_target_tokens_count


# Training args
training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size = MICRO_BATCH_SIZE,
    per_device_eval_batch_size = 1,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=0.06,
    #num_train_epochs=3,
    max_steps = 400,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type = "cosine", ## LoRA original paper uses linear
    fp16=True,
    logging_steps=50,
    evaluation_strategy="steps",
    logging_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    output_dir=lora_out_dir,
    optim = "adamw_torch",
    torch_compile = False,
    save_total_limit=2,
    load_best_model_at_end=True,
    ddp_find_unused_parameters=False if ddp else None,
)


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels

# Start trainer
trainer = transformers.Trainer(
    model=model,
    args=training_arguments,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    preprocess_logits_for_metrics = preprocess_logits_for_metrics,
)

# print("Prallel Training status: ", training_arguments.parallel_mode)
model.config.use_cache = False

# use half precision
model = to_half_precision(model)

# start training
checkpoint_dir = lora_out_dir
if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()

# Save Model
model.save_pretrained(lora_out_dir)