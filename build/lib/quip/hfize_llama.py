import argparse
import os
import glog
import torch
from transformers import LlamaTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from model.llama import LlamaForCausalLM
from lib import codebook
import time

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--quantized_path', default='checkpoints/quantized_hada_70b', type=str)
parser.add_argument('--hf_output_path', default='hfized/quantized_hada_70b', type=str)


def unpack_quip(module, saved_layer, codebook_id, codesz):
    (m, n) = saved_layer['Qidxs'].shape
    if codebook_id in codebook.cache_permute_set:
        module.Qidxs.copy_(saved_layer['Qidxs'].view(m, n // codesz,
                                                     codesz).permute(1, 0,
                                                                     2).reshape(m, n).contiguous())
    else:
        module.Qidxs.copy_(saved_layer['Qidxs'])

    if module.rank > 0:
        module.A.copy_(saved_layer['A'])
        module.B.copy_(saved_layer['B'])
    module.SU.copy_(saved_layer['SU'])
    module.SV.copy_(saved_layer['SV'])
    module.Wscale.copy_(saved_layer['Wscale'])
    if module.rescale_WH:
        module.scaleWH.copy_(saved_layer['scaleWH'])

    module.codebook_id.copy_(codebook_id)

    Q = saved_layer['Qidxs'].int() - torch.min(saved_layer['Qidxs']).int()
    cts = torch.bincount(Q.view(-1))
    cts = cts[torch.where(cts != 0)[0]]
    p = cts / cts.sum()
    ent = (p * torch.log2(1 / p)).sum()
    glog.info(f'entropy: {ent/codesz} bits per weight')


def main(args):
    assert os.path.exists(args.quantized_path)
    saved_config = torch.load(os.path.join(args.quantized_path, 'config.pt'))
    model_config = saved_config['model_config']

    if 'codebook' in model_config.quip_params:
        codebook_id = codebook.get_id(model_config.quip_params['codebook'])
        codesz = model_config.quip_params['codesz']
    else:
        codebook_id = 0
        codesz = 4

    tokenizer = LlamaTokenizer.from_pretrained(model_config._name_or_path)

    model = LlamaForCausalLM.from_pretrained(model_config._name_or_path,
                                             torch_dtype='auto',
                                             low_cpu_mem_usage=True,
                                             config=model_config).half()

    for ii in range(len(model.model.layers)):
        glog.info(f'updating layer {ii}')

        layer = model.model.layers[ii]
        cpu = torch.device('cpu')

        glog.info(f'loading layer {ii} qkv')
        saved_layer = torch.load(f'{args.quantized_path}/{ii}_qkv.pt', map_location=cpu)
        layer.self_attn.q_scale.copy_(saved_layer['W_q_scale'])
        layer.self_attn.k_scale.copy_(saved_layer['W_k_scale'])
        layer.self_attn.v_scale.copy_(saved_layer['W_v_scale'])
        unpack_quip(layer.self_attn.qkv_proj, saved_layer, codebook_id, codesz)

        glog.info(f'loading layer {ii} o')
        saved_layer = torch.load(f'{args.quantized_path}/{ii}_o.pt', map_location=cpu)
        layer.self_attn.o_scale.copy_(saved_layer['W_o_scale'])
        unpack_quip(layer.self_attn.o_proj, saved_layer, codebook_id, codesz)

        glog.info(f'loading layer {ii} up')
        saved_layer = torch.load(f'{args.quantized_path}/{ii}_up.pt', map_location=cpu)
        layer.mlp.up_scale.copy_(saved_layer['W_up_scale'])
        layer.mlp.gate_scale.copy_(saved_layer['W_gate_scale'])
        unpack_quip(layer.mlp.upgate_proj, saved_layer, codebook_id, codesz)

        glog.info(f'loading layer {ii} down')
        saved_layer = torch.load(f'{args.quantized_path}/{ii}_down.pt', map_location=cpu)
        layer.mlp.down_scale.copy_(saved_layer['W_down_scale'])

        if model_config.quip_params['outlier_channel_split']:
            layer.mlp.down_proj.ocs_dupe_inds.copy_(torch.tensor(saved_layer['ocs_dupe_inds']))

        unpack_quip(layer.mlp.down_proj, saved_layer, codebook_id, codesz)

    glog.info(f'saving model...')
    model.save_pretrained(args.hf_output_path, safe_serialization=True)

    del model

    model = LlamaForCausalLM.from_pretrained(args.hf_output_path,
                                             torch_dtype='auto',
                                             low_cpu_mem_usage=True,
                                             device_map='auto').half()

    glog.info('successfully loaded hfized model')

    glog.info('generating some text...')

    start = time.time()
    prompt = 'It is a truth universally acknowledged that'
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(input_ids=inputs['input_ids'].cuda(),
                             attention_mask=inputs['attention_mask'].cuda(),
                             max_new_tokens=64,
                             return_dict_in_generate=True)
    token = outputs.sequences[0, :]
    output_str = tokenizer.decode(token)
    glog.info(output_str)
    glog.info(f'elapsed: {time.time() - start}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)
