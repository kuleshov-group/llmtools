"""Wraps around PEFT to use QuantLoraModel instead of regular LoraModel."""

#* Latest Integration with Peft=0.8.2, Feb 2024 *#

import peft as quant_peft
from llmtools.engine.lora.lora import ModuLoraModel
from llmtools.engine.lora.lora import LoraLayer
from llmtools.engine.lora.lora import ModuLoraConfig

# monkey patch peft to use QuantLoraModel
quant_peft.tuners.lora.LoraModel = ModuLoraModel
quant_peft.tuners.lora.LoraLayer = LoraLayer
# quant_peft.tuners.lora.LoraConfig = ModuLoraConfig #TODO: adding additional quantization_method is not working as intended.
# quant_peft.peft_model.LoraModel = ModuLoraModel

# the above works for PEFT at the time of writing this code;
# when upgrading to a newer PEFT, use this insted:
quant_peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING[
	quant_peft.utils.PeftType.LORA
] = ModuLoraModel


