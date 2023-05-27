import torch
from llmtune.llms.opt.config import OPT7B4BitConfig
from llmtune.llms.llama.config import (
    LLama7B4BitConfig, 
    LLama13B4BitConfig, 
    LLama30B4BitConfig, 
    LLama65B4BitConfig
)
from llmtune.engine.lora.config import Finetune4bConfig

# ----------------------------------------------------------------------------

# define some constants
DEV = torch.device('cuda')
LLAMA_MODELS = [
    "llama-7b-4bit", "llama-13b-4bit", "llama-30b-4bit", "llama-65b-4bit"
]
OPT_MODELS  = ["opt-6.7b-4bit"]
LLM_MODELS = LLAMA_MODELS + OPT_MODELS

# define some helpers
def get_llm_config(model):
    if model == "llama-7b-4bit":
        return LLama7B4BitConfig
    elif model == "llama-13b-4bit":
        return LLama13B4BitConfig
    elif model == "llama-30b-4bit":
        return LLama30B4BitConfig
    elif model == "llama-65b-4bit":
        return LLama65B4BitConfig
    elif model == "opt-6.7b-4bit":
        return OPT7B4BitConfig      
    else:
        raise ValueError(f"Invalid model name: {model}")

# ----------------------------------------------------------------------------

# helpers for loading finetuning configs
def get_finetune_config(args):
    return Finetune4bConfig(
        dataset=args.dataset, 
        ds_type=args.data_type, 
        lora_out_dir=args.adapter, 
        mbatch_size=args.mbatch_size,
        batch_size=args.batch_size,
        epochs=args.epochs, 
        lr=args.lr,
        cutoff_len=args.cutoff_len,
        lora_r=args.lora_r, 
        lora_alpha=args.lora_alpha, 
        lora_dropout=args.lora_dropout,
        val_set_size=args.val_set_size,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
    )
