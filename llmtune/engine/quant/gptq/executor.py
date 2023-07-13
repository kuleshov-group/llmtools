import torch
import torch.nn as nn
from llmtune.engine.quant.gptq.algorithm import GPTQ
from llmtune.engine.quant.gptq.quantizer import Quantizer
from llmtune.engine.quant.converter import make_quant
from llmtune.utils import find_layers

@torch.no_grad()
def quantize_llama(
    model, dataloader, bits, groupsize, act_order, nsamples, percdamp, 
    sym=False, true_sequential=False, nearest=False, dev='cuda'
):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), 
        dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        if true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    bits, perchannel=True, sym=sym, mse=False
                )
                
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids = position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(f'Quantizing {name} in layer {i+1}/{len(layers)}...')
                if not nearest:
                    scale,zero,g_idx = gptq[name].fasterquant(percdamp=percdamp, groupsize=groupsize, actorder=act_order)
                    quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(),scale.cpu(),zero.cpu(),g_idx.cpu())
                    gptq[name].free()
                else:
                    quantizer = Quantizer()
                    quantizer.configure(
                        bits, perchannel=True, sym=sym, mse=False
                    )
                    W = subset[name].weight.data
                    quantizer.find_params(W, weight=True)
                    quantizers['model.layers.%d.%s' % (i, name)] = (quantizer.cpu(),quantizer.scale.cpu(),quantizer.zero.cpu(),None)
                
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids = position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

def pack_llama(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [QuantLinear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name],scale,zero,g_idx = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print('Done.')
    return model