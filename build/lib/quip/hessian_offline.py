import os
import datetime
import random
import argparse
from copy import deepcopy
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import numpy
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, PreTrainedTokenizerFast, AutoTokenizer
from datasets import load_dataset

import torch.multiprocessing as mp

# import data_utils
from quip.lib import utils

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--batch_size', default=2, type=int)
parser.add_argument('--devset_size', default=256, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--base_model', default='meta-llama/Llama-2-70b-hf', type=str)
parser.add_argument('--save_path', default='hessians/llama2_70b', type=str)
parser.add_argument('--scratch_path', default=None, type=str)
parser.add_argument('--chunk_size', default=256, type=int)
parser.add_argument('--async_copy_speed', default=-1, type=int)
parser.add_argument('--act_save_rate', default=4, type=int)


def move_fn(in_q, async_copy_speed):
    # async copy to avoid slow disk
    while True:
        item = in_q.get()
        if item is None:
            return
        src, tgt = item
        if async_copy_speed > 0:
            os.system(f'rsync --bwlimit={async_copy_speed} {src} {tgt}')
        else:
            os.system(f'rsync {src} {tgt}')
        os.system(f'rm {src}')
        print(f'moved {src} to {tgt}')


def forward_layer(layer, position_ids, attention_mask, bs, device, in_q, out_q):
    torch.set_grad_enabled(False)
    layer = layer.to(device)
    position_ids = position_ids.to(device)
    attention_mask = attention_mask.to(device)
    done_qkv = utils.register_H_hook(layer.self_attn.q_proj, device)
    done_o = utils.register_H_hook(layer.self_attn.o_proj, device)
    done_up = utils.register_H_hook(layer.mlp.up_proj, device)
    done_down = utils.register_H_hook(layer.mlp.down_proj, device)

    while True:
        dev_emb = in_q.get()
        if dev_emb is None:
            layer = layer.cpu()
            position_ids = position_ids.cpu()
            attention_mask = attention_mask.cpu()
            out_q.put({'qkv': done_qkv(), 'o': done_o(), 'up': done_up(), 'down': done_down()})
            return

        assert len(dev_emb) % bs == 0
        for i in range(len(dev_emb) // bs):
            dev_emb[i * bs:(i + 1) * bs] = layer(dev_emb[i * bs:(i + 1) * bs].to(device),
                                                 position_ids=position_ids,
                                                 attention_mask=attention_mask,
                                                 use_cache=False,
                                                 output_attentions=False)[0].cpu()


def accumulate(in_q, move_q, ngpus, args, transformer_layer_index):
    Hs = {}
    mus = {}
    cts = {}

    for i in range(ngpus):
        out = in_q.get()
        if i == 0:
            for key in out:
                Hs[key] = torch.zeros(out[key][0].shape, dtype=out[key][0].dtype)
                mus[key] = torch.zeros(out[key][1].shape, dtype=out[key][1].dtype)
                cts[key] = 0
        for key in out:
            Hs[key].add_(out[key][0])
            mus[key].add_(out[key][1])
            cts[key] += out[key][2]

    keys = list(Hs.keys())

    for key in Hs:
        mus[key].div_(cts[key])
        Hs[key].div_(cts[key])
        Hs[key].addmm_(-mus[key].unsqueeze(-1), mus[key].unsqueeze(0))
        save_path = f"{args.scratch_path}/{transformer_layer_index}_{key}.pt" if args.scratch_path is not None else f"{args.save_path}/{transformer_layer_index}_{key}.pt"
        torch.save(
            {
                'flatH': utils.sym_to_flat(Hs[key].to(torch.float32)),
                'mu': mus[key].to(torch.float32),
                'n': Hs[key].shape[0],
                'ct': cts[key]
            }, save_path)
        if args.scratch_path is not None:
            move_q.put((f"{args.scratch_path}/{transformer_layer_index}_{key}.pt",
                        f"{args.save_path}/{transformer_layer_index}_{key}.pt"))

    del Hs, mus, cts, out


def main(args):
    print("loading model...")
    model = LlamaForCausalLM.from_pretrained(args.base_model,
                                             torch_dtype="auto",
                                             low_cpu_mem_usage=True)
    print("loaded model!")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    if os.path.isfile(f"{args.save_path}/dev_activations.pt"):
        print("loading cached dataset...")
        loaded_dev_activations = torch.load(f"{args.save_path}/dev_activations.pt")
        after_layer = loaded_dev_activations['after_layer']
        dev_emb = loaded_dev_activations['dev_emb']
        print(f"loaded cached dataset from {loaded_dev_activations['timestamp']}")
    else:
        print("loading dataset...")
        dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")
        devset = utils.sample_devset(dataset, tokenizer, args.devset_size, args.ctx_size)
        dev_emb = model.model.embed_tokens(devset)
        after_layer = -1
        print("loaded dataset!")

    print(f"dev_emb dtype: {dev_emb.dtype}")
    dev_emb.share_memory_()

    position_ids = torch.arange(args.ctx_size, dtype=torch.int64)[None, :] + \
        torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int64)
    attention_mask = model.model._prepare_decoder_attention_mask(
        torch.ones(args.batch_size, args.ctx_size, dtype=torch.bool),
        (args.batch_size, args.ctx_size), dev_emb[0:args.batch_size, :, :], 0)

    if args.scratch_path is not None:
        move_q = mp.Queue()
        move_p = mp.Process(target=move_fn, args=(move_q, args.async_copy_speed))
        move_p.start()
    else:
        move_q = None

    for transformer_layer_index in range(len(model.model.layers)):
        if (transformer_layer_index <= after_layer):
            print(
                f"skipping layer {transformer_layer_index} because it is before cached activations at layer {after_layer}"
            )
            continue

        transformer_layer = model.model.layers[transformer_layer_index]
        # check that there are four layers, as expected
        assert (len([m for m in transformer_layer.modules()
                     if isinstance(m, torch.nn.Linear)]) == 7)

        chunk_size = min(args.chunk_size, len(dev_emb))
        ngpus = min(torch.cuda.device_count(), len(dev_emb) // chunk_size)

        manager = mp.get_context('spawn').Manager()
        in_q = manager.Queue()
        out_q = manager.Queue()

        accumulate_proc = mp.Process(target=accumulate,
                                     args=(out_q, move_q, ngpus, args, transformer_layer_index))
        accumulate_proc.start()

        forward_procs = []
        for i in range(ngpus):
            p = mp.Process(target=forward_layer,
                           args=(transformer_layer, position_ids, attention_mask, args.batch_size,
                                 i, in_q, out_q))
            p.start()
            forward_procs.append(p)

        assert len(dev_emb) % args.batch_size == 0 and chunk_size % args.batch_size == 0
        i = 0
        while i < len(dev_emb):
            next = min(i + chunk_size, len(dev_emb))
            in_q.put(dev_emb[i:next])
            i = next

        for i in range(ngpus):
            in_q.put(None)

        for p in forward_procs:
            p.join()

        accumulate_proc.join()

        transformer_layer.cpu()
        model.model.layers[transformer_layer_index] = None
        utils.clean()

        if transformer_layer_index % args.act_save_rate == 0 or \
           transformer_layer_index == len(model.model.layers) - 1:
            if args.scratch_path is not None:
                if os.path.exists(f'{args.scratch_path}/dev_activations.pt'):
                    print('not saving layer since disk is too slow')
                else:
                    torch.save(
                        {
                            'dev_emb': dev_emb,
                            'after_layer': transformer_layer_index,
                            'timestamp': str(datetime.datetime.now())
                        }, f'{args.scratch_path}/dev_activations.pt')
                    move_q.put((f'{args.scratch_path}/dev_activations.pt',
                                f'{args.save_path}/dev_activations.pt'))
            else:
                torch.save(
                    {
                        'dev_emb': dev_emb,
                        'after_layer': transformer_layer_index,
                        'timestamp': str(datetime.datetime.now())
                    }, f'{args.save_path}/dev_activations.pt')

        print(f"done processing layer {transformer_layer_index}")

    if args.scratch_path is not None:
        move_q.put(None)
        move_p.join()


if __name__ == "__main__":
    mp.set_start_method('spawn')
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
