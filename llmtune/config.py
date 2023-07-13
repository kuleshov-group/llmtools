import torch
from llmtune.llms.opt.config import get_opt_config, OPT_MODELS
from llmtune.llms.llama.config import get_llama_config, LLAMA_MODELS
from llmtune.engine.lora.config import FinetuneConfig

# ----------------------------------------------------------------------------

# define some constants
DEV = torch.device('cuda')
LLM_MODELS = LLAMA_MODELS + OPT_MODELS

# define some helpers
def get_llm_config(model):
    if model in LLAMA_MODELS:
        return get_llama_config(model)
    elif model in OPT_MODELS:
        return get_opt_config(model)
    else:
        raise ValueError(f"Invalid model name: {model}")

# ----------------------------------------------------------------------------

# helpers for loading finetuning configs
def get_finetune_config(args):
    return FinetuneConfig(
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
