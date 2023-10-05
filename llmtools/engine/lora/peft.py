"""Wraps around PEFT to use QuantLoraModel instead of regular LoraModel."""

import peft as quant_peft
from llmtools.engine.lora.lora import QuantLoraModel

# monkey patch peft to use QuantLoraModel
quant_peft.tuners.lora.LoraModel = QuantLoraModel
quant_peft.peft_model.LoraModel = QuantLoraModel

# the above works for PEFT at the time of writing this code;
# when upgrading to a newer PEFT, use this insted:
#quant_peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING[quant_peft.utils.PeftType.LORA] = QuantLoraModel
