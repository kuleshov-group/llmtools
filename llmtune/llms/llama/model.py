import torch
import torch.nn as nn

from llmtune.utils import find_layers
from llmtune.engine.quant.converter import make_quant

def load_llama_unquantized(llm_config):
    import torch
    from transformers import LlamaForCausalLM
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = LLaMAForCausalLM.from_pretrained(
        llm_config.hf_config_name, torch_dtype='auto'
    )
    model.seqlen = 2048
    return model

def fix_tokenizer(tokenizer):
    # Fixing broken tokenizers
    special_tokens = dict()
    for token_id in range(1000):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if tokenizer.pad_token_id in (None, tokenizer.vocab_size) and "pad" in token:
            special_tokens["pad_token"] = token
        if tokenizer.bos_token_id in (None, tokenizer.vocab_size) and "<s>" in token:
            special_tokens["bos_token"] = token
        if tokenizer.eos_token_id in (None, tokenizer.vocab_size) and "</s>" in token:
            special_tokens["eos_token"] = token
        if tokenizer.unk_token_id in (None, tokenizer.vocab_size) and "unk" in token:
            special_tokens["unk_token"] = token
        if tokenizer.sep_token_id in (None, tokenizer.vocab_size) and "sep" in token:
            special_tokens["sep_token"] = token

    if tokenizer.sep_token_id in (None, tokenizer.vocab_size) and "bos_token" in special_tokens:
        special_tokens["sep_token"] = special_tokens["bos_token"]

    if tokenizer.pad_token_id in (None, tokenizer.vocab_size) and "pad_token" not in special_tokens:
        if tokenizer.unk_token_id is not None:
            special_tokens["pad_token"] = tokenizer.unk_token
        else:
            special_tokens["pad_token"] = "<|pad|>"

    if tokenizer.sep_token_id in (None, tokenizer.vocab_size) and "sep_token" not in special_tokens:
        if tokenizer.bos_token_id is not None:
            special_tokens["sep_token"] = tokenizer.bos_token
        else:
            special_tokens["sep_token"] = "<|sep|>"

    tokenizer.add_special_tokens(special_tokens)

    print("Vocab size: ", tokenizer.vocab_size)
    print("PAD: ", tokenizer.pad_token_id, tokenizer.pad_token)
    print("BOS: ", tokenizer.bos_token_id, tokenizer.bos_token)
    print("EOS: ", tokenizer.eos_token_id, tokenizer.eos_token)
    print("UNK: ", tokenizer.unk_token_id, tokenizer.unk_token)
    print("SEP: ", tokenizer.sep_token_id, tokenizer.sep_token)
    return tokenizer

def fix_model(model, tokenizer, use_resize=True):
    model.config.pad_token_id = tokenizer.pad_token_id
    assert model.config.pad_token_id is not None

    bos_candidates = (
        tokenizer.bos_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.unk_token_id
    )
    for bos_candidate in bos_candidates:
        model.config.bos_token_id = bos_candidate
        if bos_candidate is not None:
            break
    assert model.config.bos_token_id is not None
    model.config.decoder_start_token_id = model.config.bos_token_id

    eos_candidates = (tokenizer.eos_token_id, tokenizer.sep_token_id)
    for eos_candidate in eos_candidates:
        model.config.eos_token_id = eos_candidate
        if eos_candidate is not None:
            break
    assert model.config.eos_token_id is not None

    if use_resize:
        model.resize_token_embeddings(len(tokenizer))

    return model

def load_llama(llm_config, checkpoint):
    import transformers, accelerate
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
    
    with accelerate.init_empty_weights():
        config = LlamaConfig.from_pretrained(llm_config.hf_config_name)
        torch.set_default_dtype(torch.half)
        transformers.modeling_utils._init_weights = False
        torch.set_default_dtype(torch.half)

        tokenizer = LlamaTokenizer.from_pretrained(llm_config.hf_tokenizer_config)
        tokenizer = fix_tokenizer(tokenizer)
        # tokenizer.truncation_side = 'left'

        model = LlamaForCausalLM(config)
        model = fix_model(model, tokenizer, use_resize=False)
        torch.set_default_dtype(torch.float)
        model = model.eval()
        layers = find_layers(model)
        for name in ['lm_head']:
            if name in layers:
                del layers[name]
        make_quant(model, layers, llm_config.bits)
    model = accelerate.load_checkpoint_and_dispatch(
            model=model,
            checkpoint=checkpoint,
            device_map="auto",
            # device_map={'': 0},
            no_split_module_classes=["LlamaDecoderLayer"]
    )
    model.seqlen = 2048

    

    return model, tokenizer
