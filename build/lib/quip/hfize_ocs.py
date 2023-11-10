import torch
# from transformers import LlamaForCausalLM as HFLlamaModel
#from quant_fft_ocs_sp import load_quip, remove_mean, fuse_W
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
#from model.llama import LlamaForCausalLM
import latticed4
import time
import gc

torch.set_grad_enabled(False)

save_pfx = 'checkpoints/quantized-hada-70b-ocs-sp'

import accelerate


def main():
    D4_CB = latticed4.build_D4_CB()

    print("loading config...")
    # config = LlamaConfig.from_pretrained("meta-llama/Llama-2-70b-chat-hf")
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
    print("building model...")

    model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-70b-hf',
                                             torch_dtype="auto",
                                             low_cpu_mem_usage=True,
                                             device_map='auto',
                                             max_memory={
                                                 0: '26GiB',
                                                 1: '39GiB',
                                                 2: '39GiB',
                                                 3: '39GiB'
                                             })
    model = model.half()

    for ii in range(len(model.model.layers)):
        print(f"updating layer {ii}")

        transformer_layer = model.model.layers[ii]

        # model.model.layers[ii].input_layernorm = orig_model.model.layers[ii].input_layernorm
        # model.model.layers[ii].post_attention_layernorm = orig_model.model.layers[ii].post_attention_layernorm

        print(f"loading layer {ii} qkv")
        hatW = load_quip(f'{save_pfx}/{ii}_qkv.pt', D4_CB, 0).cpu()
        W_q = transformer_layer.self_attn.q_proj.weight
        W_k = transformer_layer.self_attn.k_proj.weight
        W_v = transformer_layer.self_attn.v_proj.weight
        W_q_scale = W_q.to(torch.float32).square().mean().sqrt().to(torch.float32)
        W_k_scale = W_k.to(torch.float32).square().mean().sqrt().to(torch.float32)
        W_v_scale = W_v.to(torch.float32).square().mean().sqrt().to(torch.float32)
        W_q_next = (hatW[0:(W_q.shape[0]), :].to(W_q.device) * W_q_scale).half()
        W_k_next = (hatW[(W_q.shape[0]):(W_q.shape[0] + W_k.shape[0]), :].to(W_k.device) *
                    W_k_scale).half()
        W_v_next = (hatW[(W_q.shape[0] + W_k.shape[0]):(W_q.shape[0] + W_k.shape[0] +
                                                        W_v.shape[0]), :].to(W_v.device) *
                    W_v_scale).half()
        if remove_mean:
            W_q.bias = nn.Parameter(
                (W_q.to(torch.float32) @ mu - W_q_next.to(torch.float32) @ mu).half())
            W_k.bias = nn.Parameter(
                (W_k.to(torch.float32) @ mu - W_k_next.to(torch.float32) @ mu).half())
            W_v.bias = nn.Parameter(
                (W_v.to(torch.float32) @ mu - W_v_next.to(torch.float32) @ mu).half())
        W_q.copy_(W_q_next)
        W_k.copy_(W_k_next)
        W_v.copy_(W_v_next)

        print(f"loading layer {ii} o")
        hatW = load_quip(f'{save_pfx}/{ii}_o.pt', D4_CB, 0).cpu()
        W_o = transformer_layer.self_attn.o_proj.weight
        W_o_scale = W_o.to(torch.float32).square().mean().sqrt().to(torch.float32)
        W_o_next = (hatW.to(W_o.device) * W_o_scale).half()
        if remove_mean:
            W_o.bias = nn.Parameter(
                (W_o.to(torch.float32) @ mu - W_o_next.to(torch.float32) @ mu).half())
        W_o.copy_(W_o_next)

        print(f"loading layer {ii} up")
        hatW = load_quip(f'{save_pfx}/{ii}_up.pt', D4_CB, 0).cpu()
        W_up = transformer_layer.mlp.up_proj.weight
        W_gate = transformer_layer.mlp.gate_proj.weight
        W_up_scale = W_up.to(torch.float32).square().mean().sqrt().to(torch.float32)
        W_gate_scale = W_gate.to(torch.float32).square().mean().sqrt().to(torch.float32)
        W_up_next = (hatW[0:(W_up.shape[0]), :].to(W_up.device) * W_up_scale).half()
        W_gate_next = (
            hatW[(W_up.shape[0]):(W_up.shape[0] + W_gate.shape[0]), :].to(W_gate.device) *
            W_gate_scale).half()
        if remove_mean:
            W_up.bias = nn.Parameter(
                (W_up.to(torch.float32) @ mu - W_up_next.to(torch.float32) @ mu).half())
            W_gate.bias = nn.Parameter(
                (W_gate.to(torch.float32) @ mu - W_gate_next.to(torch.float32) @ mu).half())
        W_up.copy_(W_up_next)
        W_gate.copy_(W_gate_next)

        print(f"loading layer {ii} down")
        hatW = load_quip(f'{save_pfx}/{ii}_down.pt', D4_CB, 0).cpu()
        W_down = transformer_layer.mlp.down_proj.weight
        W_down_scale = W_down.to(torch.float32).square().mean().sqrt().to(torch.float32)
        W_down_next = (hatW.to(W_down.device) * W_down_scale).half()

        ocs_inds = torch.load(f'{save_pfx}/{ii}_ocs_inds.pt')
        W_down_next = fuse_W(W_down_next, ocs_inds)

        if remove_mean:
            W_down.bias = nn.Parameter(
                (W_down.to(torch.float32) @ mu - W_down_next.to(torch.float32) @ mu).half())
        W_down.copy_(W_down_next)

        gc.collect()
        torch.cuda.empty_cache()

    print(f"saving model...")
    model.save_pretrained('./llama_hada_d4_70b', safe_serialization=True)

    print("generating some text...")

    start = time.time()

    prompt = 'It is a truth universally acknowledged that'
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(input_ids=inputs['input_ids'].cuda(),
                             attention_mask=inputs['attention_mask'].cuda(),
                             max_new_tokens=64,
                             return_dict_in_generate=True)
    token = outputs.sequences[0, :]
    output_str = tokenizer.decode(token)
    print(output_str)

    end = time.time()

    print(f"elapsed: {end - start}")


if __name__ == "__main__":
    main()
