import torch
import torch.nn as nn

from llmtune.utils import find_layers
from llmtune.engine.quant.converter import make_quant

def load_llama2_unquantized(llm_config):
    import torch
    from transformers import LlamaForCausalLM
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = LlamaForCausalLM.from_pretrained(
        llm_config.base_config.name_or_path, torch_dtype='auto'
    )
    return model

def load_llama2_quantized(llm_config, quantized_weights_path):
    import transformers, accelerate
    from transformers import LlamaConfig, LlamaForCausalLM
    
    with accelerate.init_empty_weights():
        config = LlamaConfig.from_pretrained(
            llm_config.base_config.name_or_path
        )
        torch.set_default_dtype(torch.half)
        transformers.modeling_utils._init_weights = False
        torch.set_default_dtype(torch.half)
        model = LlamaForCausalLM(config)
        torch.set_default_dtype(torch.float)
        model = model.eval()
        layers = find_layers(model)
        for name in ['lm_head']:
            if name in layers:
                del layers[name]
        make_quant(
            model, layers, llm_config.quant_config.bits, 
            groupsize=llm_config.quant_config.groupsize
        )
    model = accelerate.load_checkpoint_and_dispatch(
            model=model,
            checkpoint=quantized_weights_path,
            device_map="auto",
            # device_map={'': 0},
            no_split_module_classes=["LlamaDecoderLayer"]
    )
    return model

def load_llama2(llm_config, quantized_weights_path):
    if quantized_weights_path is None:
        model = load_llama2_unquantized(llm_config)
    else:
        model = load_llama2_quantized(
            llm_config, quantized_weights_path
        )
    model.seqlen = 2048
    return model

def load_llama2_tokenizer(name_or_path):
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        name_or_path
    )
    tokenizer.truncation_side = 'left'
    return tokenizer
