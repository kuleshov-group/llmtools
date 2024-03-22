"""Wraps around PEFT to use QuantLoraModel instead of regular LoraModel."""

#* Latest Integration with Peft=0.8.2, Feb 2024 *#
import peft as quant_peft

#* monkey patch peft to use QuantLoraModel *#
# Unused: Already imported in ModuLoRA PeftModel.

# from llmtools.engine.lora.lora import ModuLoraModel
# from llmtools.engine.lora.lora import LoraLayer
# from llmtools.engine.lora.lora import ModuLoraConfig

# the above works for PEFT at the time of writing this code;
# when upgrading to a newer PEFT, use this insted:
# quant_peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING[
# 	quant_peft.utils.PeftType.LORA
# ] = ModuLoraModel

#* monkey patch peft to use set_peft_model_state_dict from LLM *#
from llmtools.engine.lora.peft_model import PeftModel
from llmtools.engine.lora.peft_model import PeftModelForCausalLM 
from llmtools.engine.lora.mapping import get_peft_model

quant_peft.peft_model.PeftModel = PeftModel
quant_peft.PeftModel = PeftModel
quant_peft.peft_model.PeftModelForCausalLM = PeftModelForCausalLM


quant_peft.get_peft_model = get_peft_model
