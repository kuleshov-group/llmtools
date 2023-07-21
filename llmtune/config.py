import torch
from llmtune.llms.config import AutoConfig
from llmtune.llms.opt.config import OPT_MODELS
from llmtune.llms.llama.config import LLAMA_MODELS
from llmtune.engine.lora.config import FinetuneConfig
from llmtune.engine.quant.config import QuantConfig

# ----------------------------------------------------------------------------

# define some constants
DEV = torch.device('cuda')
LLM_MODELS = LLAMA_MODELS + OPT_MODELS

# ----------------------------------------------------------------------------

# helpers for loading configs
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

def get_quant_config(args):
    return QuantConfig(
        dataset=args.dataset,
        bits=args.bits,
        nsamples=args.nsamples,
        groupsize=args.groupsize,
        act_order=args.act_order,
        percdamp=args.percdamp,
        seed=args.seed,
        nearest=args.nearest,
        save=args.save,
    )

def get_llm_config(model_name_or_path):
    return AutoConfig(model_name_or_path)