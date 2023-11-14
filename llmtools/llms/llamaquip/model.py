import glog
import os
import json
from transformers import LlamaTokenizer
from quip.model.llama import LlamaForCausalLM

def load_llama_quip(hf_path):
    model = LlamaForCausalLM.from_pretrained(hf_path,
                                            torch_dtype='auto',
                                            low_cpu_mem_usage=True,
                                            device_map='auto') 

    with open(os.path.join(hf_path, 'config.json')) as f:
        _config = json.load(f)
    _name_or_path = _config['_name_or_path']
    tokenizer = LlamaTokenizer.from_pretrained(_name_or_path, add_eos_token=True) # Append EOS token to end of the sentence

    glog.info('loaded model!')
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, _config