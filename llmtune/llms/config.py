import os
import json
from dataclasses import dataclass
from enum import Enum
from transformers.utils.hub import PushToHubMixin, cached_file, create_repo

class ModelType(Enum):
    LLAMA = 'llama'
    OPT = 'opt'

@dataclass
class LLMConfig(PushToHubMixin):
    name: str # model name e.g., llama-65b-4bit-g64
    model_type: ModelType
    hf_config_name: str
    hf_tokenizer_config: str
    bits: int
    groupsize: int = -1

    def save_pretrained(self, save_dir: str, **kwargs):
        config_path = join(save_dir, "llmtune_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, save_dir: str, **kwargs):
        config_filename = "llmtune_config.json"
        if os.path.isdir(save_dir): 
            config_path = join(save_dir, config_filename)
        else:
            config_path = cached_file(save_dir, config_filename)
        with open(config_path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))
                
    def to_dict(self):
        return {
            "name": self.name,
            "model_type": self.model_type,
            "hf_config_name": self.hf_config_name,
            "hf_tokenizer_config": self.hf_tokenizer_config,
            "bits": self.bits,
            "groupsize": self.groupsize,
        }