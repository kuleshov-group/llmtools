#* Modulora Integration: OPTQ *#
import os
import json
from dataclasses import dataclass
from transformers.utils.hub import PushToHubMixin, cached_file

@dataclass
class QuantConfig(PushToHubMixin):
    dataset: str
    bits: int
    nsamples: int
    groupsize: int
    act_order: bool
    percdamp: float
    seed: int
    nearest: bool
    save: str

    def save_pretrained(self, save_dir: str, **kwargs):
        config_path = os.path.join(save_dir, "quant_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, save_dir: str, **kwargs):
        config_filename = "quant_config.json"
        if os.path.isdir(save_dir): 
            config_path = os.path.join(save_dir, config_filename)
        else:
            config_path = cached_file(save_dir, config_filename)
        with open(config_path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))
                
    def to_dict(self):
        return {
            'dataset': self.dataset,
            'bits': self.bits,
            'nsamples': self.nsamples,
            'groupsize': self.groupsize,
            'act_order': self.act_order,
            'percdamp': self.percdamp,
            'seed': self.seed,
            'nearest': self.nearest,
            'save': self.save,
        }