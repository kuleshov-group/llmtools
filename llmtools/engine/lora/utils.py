# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
# Edite by Volodymyr Kuleshov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#* Peft Integration: no longer used *#

import torch

import enum


class PeftType(str, enum.Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    ADALORA = "ADALORA"
    ADAPTION_PROMPT = "ADAPTION_PROMPT"
    IA3 = "IA3"



def prepare_model_for_quip_training(
        model,
):
    r"""
    This method wraps the quip# 
    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """  
    loaded_in_quip = True
    return model



def prepare_model_for_int4_training(
    model, output_embedding_layer_name="lm_head", use_gradient_checkpointing=False, layer_norm_names=["layer_norm"]
):
    r"""
    This method wrapps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32
    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    # loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
    loaded_in_4bit = True

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

        if loaded_in_4bit:
            # cast layer norm in fp32 for stability for 4bit models
            if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
                param.data = param.data.to(torch.float32)

    if loaded_in_4bit and use_gradient_checkpointing:
        raise NotImplementedError()

    if hasattr(model, output_embedding_layer_name):
        output_embedding_layer = getattr(model, output_embedding_layer_name)
        input_dtype = output_embedding_layer.weight.dtype

        class CastOutputToFloat(torch.nn.Sequential):
            r"""
            Manually cast to the expected dtype of the lm_head as sometimes there is a final layer norm that is casted
            in fp32
            """

            def forward(self, x):
                return super().forward(x.to(input_dtype)).to(torch.float32)

        setattr(model, output_embedding_layer_name, CastOutputToFloat(output_embedding_layer))

    return model


def set_peft_model_state_dict(model, peft_model_state_dict, adapter_name="default"):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
    """
    config = model.peft_config[adapter_name]
    state_dict = {}
    if getattr(model, "modules_to_save", None) is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(module_name, f"{module_name}.modules_to_save.{adapter_name}")
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    if config.peft_type in (PeftType.LORA, PeftType.ADALORA, PeftType.IA3):
        peft_model_state_dict = {}
        parameter_prefix = "ia3_" if config.peft_type == PeftType.IA3 else "lora_"
        for k, v in state_dict.items():
            if parameter_prefix in k:
                suffix = k.split(parameter_prefix)[1]
                if "." in suffix:
                    suffix_to_replace = ".".join(suffix.split(".")[1:])
                    k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
                else:
                    k = f"{k}.{adapter_name}"
                peft_model_state_dict[k] = v
            else:
                peft_model_state_dict[k] = v
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)
    elif config.is_prompt_learning or config.peft_type == PeftType.ADAPTION_PROMPT:
        peft_model_state_dict = state_dict
    else:
        raise NotImplementedError
    
    #* ModuLoRA TODO: Rename peft_model_state_dict keys() for appropriate weights reassignment *# 
    peft_model_state_dict = rename_modulora_adapters_in_dict(peft_model_state_dict)
    
    load_result = model.load_state_dict(peft_model_state_dict, strict=False)
    if config.is_prompt_learning:
        model.prompt_encoder[adapter_name].embedding.load_state_dict(
            {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
        )
    return load_result


def rename_modulora_adapters_in_dict(original_dict, adapter_name="default"):
    updated_dict = original_dict.copy() 
    
    for key in list(original_dict.keys()):
        segments = key.split('.')
        
        for i, segment in enumerate(segments):
            if "lora_A_" in segment or "lora_B_" in segment:
                prefix, num = segment.rsplit('_', 1)
                if i+1 < len(segments) and segments[i+1] == adapter_name:
                    segments[i] = prefix  # Update the current segment
                    segments[i+1] = f'{adapter_name}_{num}'  # Update the 'default' segment
                    
                    # Construct the new key and update the dictionary
                    new_key = '.'.join(segments)
                    updated_dict[new_key] = updated_dict.pop(key)  # Replace key in the dict
                    
                    break  # Assuming only one modification per key
    
    return updated_dict