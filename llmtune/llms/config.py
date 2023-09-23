import os
import json
from enum import Enum
from transformers import PretrainedConfig, AutoConfig
from transformers.utils.hub import PushToHubMixin, cached_file
from llmtune.engine.quant.config import QuantConfig

class LLMType(Enum):
    LLAMA = 'llama'
    LLAMA2 = 'llama2'
    OPT = 'opt'
    BLOOM = 'bloom'

class AutoLLMConfig(PretrainedConfig,PushToHubMixin):
    def __init__(
        self, 
        base_config: PretrainedConfig, 
        quant_config: QuantConfig = None
    ):
        self.base_config = base_config
        self.quant_config = None
        if quant_config is not None:
            self.quant_config = quant_config

    @property
    def is_quantized(self):
        return self.quant_config is not None

    def set_quant_config(self, quant_config):
        if self.quant_config is not None:
            raise RuntimeError('quant_config already set')
        self.quant_config = quant_config

    @property
    def model_type(self):
        return self.base_config.model_type

    def save_pretrained(self, save_dir: str, **kwargs):
        self.base_config.save_pretrained(save_dir, **kwargs)
        if self.is_quantized:
            self.quant_config.save_pretrained(save_dir, **kwargs)

    @classmethod
    def from_pretrained(cls, save_dir: str):
        # load config
        base_config = AutoConfig.from_pretrained(save_dir)

        # check if quantized model and config are available
        try:
            quant_config = (
                QuantConfig.from_pretrained(save_dir)
            )
        except:
            quant_config = None

        # check if it's a valid model
        if base_config.model_type not in [e.value for e in LLMType]:
            raise NotImplementedError(
                f"Model type {base_config.model_type} currently not supported"
            )

        return cls(base_config, quant_config)
