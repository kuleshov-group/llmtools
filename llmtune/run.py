import argparse
from llmtune.config import LLM_MODELS
import torch
import torch.nn as nn
# ----------------------------------------------------------------------------

def make_parser():
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=lambda args: parser.print_help())
    subparsers = parser.add_subparsers(title='Commands')

    # generate

    gen_parser = subparsers.add_parser('generate')
    gen_parser.set_defaults(func=generate)

    gen_parser.add_argument('--model', choices=LLM_MODELS, required=True,
        help='Type of model to load')
    gen_parser.add_argument('--weights', type=str, required=True,
        help='Path to the base model weights.')
    gen_parser.add_argument('--adapter', type=str, required=False,
        help='Path to the folder with the Lora adapter.')
    gen_parser.add_argument('--prompt', type=str, default='',
        help='Text used to initialize generation')
    gen_parser.add_argument('--instruction', type=str, default='',
        help='Instruction for an alpaca-style model')    
    gen_parser.add_argument('--min-length', type=int, default=10, 
        help='Minimum length of the sequence to be generated.')
    gen_parser.add_argument('--max-length', type=int, default=200,
        help='Maximum length of the sequence to be generated.')
    gen_parser.add_argument('--top_p', type=float, default=.95,
        help='Top p sampling parameter.')
    gen_parser.add_argument('--top_k', type=int, default=50,
        help='Top p sampling parameter.')
    gen_parser.add_argument('--temperature', type=float, default=1.0,
        help='Sampling temperature.')

    # quantize

    quant_parser = subparsers.add_parser('quantize')
    quant_parser.set_defaults(func=quantize)

    quant_parser.add_argument('--model', choices=LLM_MODELS, required=True,
        help='Type of model to load')
    quant_parser.add_argument('--weights', type=str, required=True,
        help='Path to the saved model weights.')
    quant_parser.add_argument('dataset', type=str, 
        choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.')
    quant_parser.add_argument('--seed', type=int, default=0, 
        help='Seed for sampling the calibration data.')
    quant_parser.add_argument('--nsamples', type=int, default=128,
        help='Number of calibration data samples.')
    quant_parser.add_argument('--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.')
    quant_parser.add_argument('--wbits', type=int, default=4, 
        choices=[2, 3, 4, 8], help='#bits to use for quantization.')
    quant_parser.add_argument('--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.')
    quant_parser.add_argument('--save', type=str, default='',
        help='Save quantized checkpoint under this name.')

    # download

    dl_parser = subparsers.add_parser('download')
    dl_parser.set_defaults(func=download)

    dl_parser.add_argument('--model', choices=LLM_MODELS, required=True,
        help='Type of model to load')
    dl_parser.add_argument('--weights', type=str, default='./weights.pt',
        help='File where weights will be stored')

    # finetune

    tune_parser = subparsers.add_parser('finetune')
    tune_parser.set_defaults(func=finetune)

    # Config args group
    tune_parser.add_argument('--model', choices=LLM_MODELS, required=True,
        help='Type of model to load')
    tune_parser.add_argument('--weights', type=str, required=True,
        help='Path to the model weights.')
    tune_parser.add_argument("--data-type", choices=["alpaca", "gpt4all"],
        help="Dataset format", default="alpaca")
    tune_parser.add_argument("--dataset", required=False,
        help="Path to local dataset file.")
    tune_parser.add_argument('--adapter', type=str, required=False,
        help='Path to Lora adapter folder (also holds checkpoints)')

    # Training args group
    tune_parser.add_argument("--mbatch_size", default=1, type=int, 
        help="Micro-batch size. ")
    tune_parser.add_argument("--batch_size", default=2, type=int, 
        help="Batch size. ")
    tune_parser.add_argument("--epochs", default=3, type=int, 
        help="Epochs. ")
    tune_parser.add_argument("--lr", default=2e-4, type=float, 
        help="Learning rate. ")
    tune_parser.add_argument("--cutoff_len", default=256, type=int, 
        help="")
    tune_parser.add_argument("--lora_r", default=8, type=int, 
        help="")
    tune_parser.add_argument("--lora_alpha", default=16, type=int, 
        help="")
    tune_parser.add_argument("--lora_dropout", default=0.05, type=float, 
        help="")
    tune_parser.add_argument("--val_set_size", default=0.2, type=float, 
        help="Validation set size. ")
    tune_parser.add_argument("--warmup_steps", default=50, type=int, 
        help="")
    tune_parser.add_argument("--save_steps", default=50, type=int, 
        help="")
    tune_parser.add_argument("--save_total_limit", default=3, type=int, 
        help="")
    tune_parser.add_argument("--logging_steps", default=10, type=int, 
        help="")
    tune_parser.add_argument("--resume_checkpoint", action="store_true", 
        help="Resume from checkpoint.")

    return parser

# ----------------------------------------------------------------------------

def main():
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)

def generate(args):
    import llmtune.executor as llmtune
    llm, tokenizer = llmtune.load_llm(args.model, args.weights)
    if args.adapter is not None:
        llm = llmtune.load_adapter(llm, adapter_path=args.adapter)
    if args.prompt and args.instruction:
        raise Exception('Cannot specify both prompt and instruction')
    if args.instruction:
        from llmtune.engine.data.alpaca import make_prompt
        prompt = make_prompt(args.instruction, input_="")
    else:
        prompt = args.prompt

    output = llmtune.generate(
        llm, 
        tokenizer, 
        prompt, 
        args.min_length, 
        args.max_length, 
        args.temperature,        
        args.top_k, 
        args.top_p, 
    )

    if args.instruction:
        from llmtune.engine.data.alpaca import make_output
        output = make_output(output)

    print(output)

def download(args):
    from llmtune.config import get_llm_config
    from llmtune.utils import download_file
    llm_config = get_llm_config(args.model)
    if not llm_config.weights_url:
        raise Exception(f"Downloading {args.model} is not supported")
    download_file(llm_config.weights_url, args.weights)

def finetune(args):
    import torch
    from llmtune.executor import load_llm
    llm, tokenizer = load_llm(args.model, args.weights)
    from llmtune.config import get_finetune_config
    finetune_config = get_finetune_config(args)
    from llmtune.executor import finetune
    finetune(llm, tokenizer, finetune_config)

def quantize(args):
    from llmtune.config import get_llm_config
    import llmtune.executor as llmtune
    llm_config = get_llm_config(args.model)
    output = llmtune.quantize(
        llm_config, 
        args.dataset, 
        args.nsamples, 
        args.wbits, 
        args.groupsize, 
        args.percdamp, 
        args.seed,  
        args.weights     
    )

if __name__ == '__main__':
    main()    