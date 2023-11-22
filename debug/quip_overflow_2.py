import os
import json
import argparse
import torch
import datasets
from transformers import LlamaTokenizer  #, LlamaForCausalLM
from quip.model.llama import LlamaForCausalLM
import random
import glog

from quip.lib.utils import LMEvalAdaptor
from lm_eval import evaluator, tasks

from tokenizers.processors import TemplateProcessing


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='/share/kuleshov/jy928/two_bit_quant/quip/hfized/d4_7b', type=str)
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
        tokenizer = LlamaTokenizer.from_pretrained(_name_or_path, add_eos_token=True) # Append EOS token to end of the sentence

    glog.info('loaded model!')
    tokenizer.pad_token = tokenizer.eos_token

    lm_eval_model = LMEvalAdaptor(_name_or_path, model, tokenizer, args.batch_size)
    sample_question = "Write a well-thought out recipe for a new blueberry lasagna dish: "
    #sample_question_tokens = lm_eval_model.tok_encode(sample_question)

    sample_question_tokens = tokenizer(sample_question, return_tensors='pt')
    sample_question_tokens = sample_question_tokens['input_ids'].to(lm_eval_model.device)
    print(sample_question_tokens)

    sample_question_results_tokens = lm_eval_model._model_generate(sample_question_tokens, max_length=128, eos_token_id=2)
    print(sample_question_results_tokens) #? Produce unkonwn tokens all the way.
    breakpoint()

    # the model not stopping does not mean that it is not adding the eos_token but rather not predicting it.
    sample_question_results = lm_eval_model.tok_decode(sample_question_results_tokens.squeeze())
    print(sample_question_results)

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
