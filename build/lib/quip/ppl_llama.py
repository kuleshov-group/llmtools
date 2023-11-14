import os
import json
import argparse
import torch
import datasets
from transformers import LlamaTokenizer  #, LlamaForCausalLM
from quip.model.llama import LlamaForCausalLM
import random
import glog

from tqdm import tqdm

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='hfized/quantized_hada_70b', type=str)
parser.add_argument('--seqlen', default=4096, type=int)
parser.add_argument('--dataset', type=str, choices=['wiki', 'ptb', 'c4'])


def main(args):
    if 'meta-llama' in args.hf_path:
        import transformers
        model = transformers.LlamaForCausalLM.from_pretrained(args.hf_path,
                                                              torch_dtype='auto',
                                                              low_cpu_mem_usage=True,
                                                              device_map='auto').half()
        tokenizer = LlamaTokenizer.from_pretrained(args.hf_path)
    else:
        with open(os.path.join(args.hf_path, 'config.json')) as f:
            _config = json.load(f)
        model = LlamaForCausalLM.from_pretrained(args.hf_path,
                                                 torch_dtype='auto',
                                                 low_cpu_mem_usage=True,
                                                 device_map='auto').half()
        tokenizer = LlamaTokenizer.from_pretrained(_config['_name_or_path'])
    glog.info('loaded model!')
    tokenizer.pad_token = tokenizer.eos_token

    if args.dataset == 'c4':
        valdata = datasets.load_dataset(
            'allenai/c4',
            'allenai--c4',
            data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
            split='validation',
            use_auth_token=False)
        data_key = 'text'
    elif args.dataset == 'wiki':
        valdata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        data_key = 'text'
    elif args.dataset == 'ptb':
        valdata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='test')
        data_key = 'sentence'
    else:
        raise NotImplementedError

    valenc = []

    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i][data_key], return_tensors='pt')
            if tmp.input_ids.shape[1] >= args.seqlen:
                break
        # i = random.randint(0, tmp.input_ids.shape[1] - args.seqlen - 1)
        i = random.randint(0, tmp.input_ids.shape[1] - args.seqlen)
        j = i + args.seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    input_tok = valenc
    nsamples = input_tok.numel() // args.seqlen

    input_tok = input_tok[0, :(args.seqlen * nsamples)].view(nsamples, args.seqlen)

    glog.info(input_tok.shape)

    loss_fct = torch.nn.CrossEntropyLoss().cuda()
    acc_loss = 0.0
    for ii in tqdm(range(nsamples)):
        input = input_tok[ii, :].cuda().view(1, -1)
        output = model(input, use_cache=False, output_hidden_states=False,
                       output_attentions=False)[0]
        shift_logits = output[:, :-1, :].contiguous()
        shift_labels = input[:, 1:]
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        acc_loss += loss.item()
        if (((ii + 1) % 100) == 0):
            avg_loss = acc_loss / (ii + 1)
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            glog.info(f'perplexity @{ii+1}: {ppl}')
    avg_loss = acc_loss / nsamples

    ppl = torch.exp(torch.tensor(avg_loss)).item()
    glog.info(f'perplexity: {ppl}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
