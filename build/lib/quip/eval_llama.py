import os
import json
import argparse
import torch
import datasets
from transformers import LlamaTokenizer  #, LlamaForCausalLM
from model.llama import LlamaForCausalLM
import random
import glog

from lib.utils import LMEvalAdaptor
from lm_eval import evaluator, tasks

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='hfized/quantized_hada_70b', type=str)
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--tasks", type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument('--num_fewshot', type=int, default=0)


def main(args):
    if 'meta-llama' in args.hf_path:
        import transformers
        model = transformers.LlamaForCausalLM.from_pretrained(args.hf_path,
                                                              torch_dtype='auto',
                                                              low_cpu_mem_usage=True,
                                                              device_map='auto')
        _name_or_path = model.config._name_or_path
        tokenizer = LlamaTokenizer.from_pretrained(args.hf_path)
    else:
        model = LlamaForCausalLM.from_pretrained(args.hf_path,
                                                 torch_dtype='auto',
                                                 low_cpu_mem_usage=True,
                                                 device_map='auto')
        with open(os.path.join(args.hf_path, 'config.json')) as f:
            _config = json.load(f)
        _name_or_path = _config['_name_or_path']
        tokenizer = LlamaTokenizer.from_pretrained(_name_or_path)

    glog.info('loaded model!')
    tokenizer.pad_token = tokenizer.eos_token

    task_names = args.tasks.split(",")

    lm_eval_model = LMEvalAdaptor(_name_or_path, model, tokenizer, args.batch_size)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        batch_size=args.batch_size,
        no_cache=True,
        num_fewshot=args.num_fewshot,
    )

    print(evaluator.make_table(results))

    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        # otherwise cannot save
        results["config"]["model"] = args.hf_path
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
