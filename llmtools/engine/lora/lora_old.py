# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
# Edited by Volodymyr Kuleshov
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
import math
import torch
import torch.nn as nn
from torch.nn import Conv1d
import warnings

from llmtools.engine.inference.modules import QuantLinear
from llmtools.engine.lora.peft import quant_peft
#* QUIP Quant Linear Layer path *#
from quip_sharp.lib.linear.quantized_linear import QuantizedLinear
from quip_sharp.lib.linear.fused_quantized_linear import FusedQuantizedLinear

# hacky way to do imports for now
LoraLayer = quant_peft.tuners.lora.LoraLayer


#* Latest integration with peft=0.8.2 *#
class QuantLoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, model, config, adapter_name= "default"):
        super().__init__()
        self.peft_config = config
        self.model = model

        #* Latest Peft Integration *#
        self.active_adapter = adapter_name
        self.self.model.peft_config = self.peft_config

        self._find_and_replace()
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward
        

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                if hasattr(target, "bias"): #* QUIP *#
                    bias = target.bias is not None
                if isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = nn.Linear(target.in_features, target.out_features, bias=bias, **kwargs)
                elif isinstance(target, QuantLinear) and self.peft_config.enable_lora is None:
                    new_module = LinearQuantLt(
                        # target.bits,
                        target.in_features, 
                        target.out_features, 
                        target.groupsize,
                        bias=bias, 
                        is_cuda=target.is_cuda,
                        **kwargs
                    )
                #* QUIP Implementation *#
                elif isinstance (target, FusedQuantizedLinear) and self.peft_config.enable_lora is None:
                    new_module = FusedLinearQuantLtQuip(
                        target.fuse_dim,
                        target.fuse_sizes,
                        # target.bits,
                        target.in_features, 
                        target.out_features, 
                        target.codesz,
                        target.packsz,
                        target.pack_out,
                        target.idx_dtype,
                        target.codebook_version,
                        target.rank,
                        target.rescale_WH,
                        **kwargs
                    )
                #* QUIP Implementation *#
                elif isinstance(target, QuantizedLinear) and self.peft_config.enable_lora is None:
                    new_module = LinearQuantLtQuip(
                        target.in_features, 
                        target.out_features, 
                        target.codesz,
                        target.packsz,
                        target.pack_out,
                        target.idx_dtype,
                        target.codebook_version,
                        target.rank,
                        target.rescale_WH,
                        **kwargs
                    )
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        ## QUIP Implementation
        if isinstance(old_module, FusedQuantizedLinear) and isinstance(new_module, FusedLinearQuantLtQuip):
            #* Fused Linear Layer *#
            new_module.fuse_scales = old_module.fuse_scales
            new_module.fuse_dim = old_module.fuse_dim
            new_module.fuse_sizes = old_module.fuse_sizes
            new_module.n = old_module.n

            new_module.Qidxs = old_module.Qidxs
            new_module.codebook_id = old_module.codebook_id
            new_module.SU = old_module.SU
            new_module.SV = old_module.SV
            new_module.Wscale = old_module.Wscale

            new_module.rank = old_module.rank
            new_module.A = old_module.A
            new_module.B = old_module.B
            new_module.rescale_WH = old_module.rescale_WH
            new_module.scaleWH = old_module.scaleWH

            new_module.codesz = old_module.codesz
            new_module.idx_dtype = old_module.idx_dtype

            new_module.packsz = old_module.packsz
            new_module.pack_out = old_module.pack_out
            new_module.codebook_version = old_module.codebook_version

            #? Understand this ?#
            if getattr(old_module, "state", None) is not None:
                new_module.state = old_module.state
                new_module.to(old_module.Qidxs.device) 
            
            # dispatch to correct device
            for name, module in new_module.named_modules():
                if "lora_" in name:
                    module.to(old_module.Qidxs.device)
        elif isinstance(old_module, QuantizedLinear) and isinstance(new_module, LinearQuantLtQuip):
            #new_module.D4_CB = old_module.D4_CB #? D4_CB is our quantzied weighgt
            new_module.Qidxs = old_module.Qidxs
            new_module.codebook_id = old_module.codebook_id
            new_module.SU = old_module.SU
            new_module.SV = old_module.SV
            new_module.Wscale = old_module.Wscale

            new_module.rank = old_module.rank
            new_module.A = old_module.A
            new_module.B = old_module.B
            new_module.rescale_WH = old_module.rescale_WH
            new_module.scaleWH = old_module.scaleWH

            new_module.codesz = old_module.codesz
            new_module.idx_dtype = old_module.idx_dtype

            new_module.packsz = old_module.packsz
            new_module.pack_out = old_module.pack_out
            new_module.codebook_version = old_module.codebook_version

            #? Understand this ?#
            if getattr(old_module, "state", None) is not None:
                new_module.state = old_module.state
                new_module.to(old_module.Qidxs.device) 
            
            # dispatch to correct device
            for name, module in new_module.named_modules():
                if "lora_" in name:
                    module.to(old_module.Qidxs.device) ## TODO: Not sure about this. QUIP doesn't store Qweights equialent to LORA.

        elif isinstance(old_module, QuantLinear) and isinstance(new_module, LinearQuantLt):
            new_module.qweight = old_module.qweight
            new_module.scales = old_module.scales
            new_module.qzeros = old_module.qzeros
            new_module.g_idx = old_module.g_idx
            new_module.bias = old_module.bias
            new_module.groupsize = old_module.groupsize
            new_module.maxq = old_module.maxq
            new_module.bits = old_module.bits
            new_module.wf = old_module.wf
            new_module.is_cuda = old_module.is_cuda
            if getattr(old_module, "state", None) is not None:
                new_module.state = old_module.state
                new_module.to(old_module.qweight.device)

            # dispatch to correct device
            for name, module in new_module.named_modules():
                if "lora_" in name:
                    module.to(old_module.qweight.device)
        else:
            new_module.weight = old_module.weight
            if old_module.bias is not None:
                new_module.bias = old_module.bias
            if getattr(old_module, "state", None) is not None:
                new_module.state = old_module.state
                new_module.to(old_module.weight.device)

            # dispatch to correct device
            for name, module in new_module.named_modules():
                if "lora_" in name:
                    module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        # breakpoint()
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


class LinearQuantLt(QuantLinear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            bits,
            in_features,
            out_features,
            groupsize,
            bias=False,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            is_cuda=True,
            **kwargs,
    ):
        QuantLinear.__init__(
            self,
            bits=bits,
            in_features=in_features,
            out_features=out_features,
            groupsize=groupsize,
            bias=bias,
            is_cuda=is_cuda,
        )
        LoraLayer.__init__(
            self, 
            r=r, 
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout, 
            merge_weights=False
        )
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.qweight.requires_grad = False
            self.scales.requires_grad = False
            self.qzeros.requires_grad = False
            self.g_idx.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor):
        result = super().forward(x)

        if self.disable_adapters:
            return result
        elif self.r > 0:
            if not torch.is_autocast_enabled():
                expected_dtype = result.dtype

                if x.dtype != torch.float32:
                    x = x.float()
                output = self.lora_B(self.lora_A(self.lora_dropout(x))).to(expected_dtype) * self.scaling
                result += output
            else:
                output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
                result += output
            # print("Lora A weights: ", self.lora_A.weight[0])
            # print("Lora B weights: ", self.lora_B.weight[0])
        return result

def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError



##* Quip Implementation *##
class LinearQuantLtQuip(QuantizedLinear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            in_features,
            out_features,
            codesz,
            packsz,
            pack_out,
            idx_dtype,
            codebook_version,
            lora_rank,
            rescale_WH,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            is_cuda=True,
            **kwargs,
    ):
        QuantizedLinear.__init__(self,
                                in_features=in_features,
                                out_features=out_features,
                                codesz=codesz,
                                packsz=packsz,
                                pack_out=pack_out,
                                idx_dtype=idx_dtype,
                                codebook_version=codebook_version,
                                outlier_channel_split=False,
                                rank=lora_rank,
                                rescale_WH=rescale_WH)
        LoraLayer.__init__(
            self, 
            r=r, 
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout, 
            merge_weights=False
        )
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Linear(in_features, r, bias=False)
            self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r
            # Freezing the tensors except for LoRA parameters
            ## TODO: Change the grad_requirement according to QUIP configuraion ##
            ## ? The varaibles are simply defined in the forward function?
            ## ? Refers to QuantizedLinear() module in QUIP.Lib.Linear.quantized_linear.py ? ##
            if hasattr(self, "codebook_class"): self.codebook_class.requires_grad = False #* no codebook_class in init *#
            # self.D4_CB.requires_grad = False #* no codebook_class in init *#
            self.Qidxs.requires_grad = False
            self.SU.requires_grad = False
            self.SV.requires_grad = False
            self.Wscale.requires_grad = False
            if self.A is not None: self.A.requires_grad = False
            if self.B is not None: self.B.requires_grad = False
            if self.scaleWH is not None: self.scaleWH.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor):
        # breakpoint()
        result = super().forward(x)

        if self.disable_adapters:
            return result
        elif self.r > 0:
            if not torch.is_autocast_enabled():
                expected_dtype = result.dtype

                if x.dtype != torch.float32:
                    x = x.float()
                output = self.lora_B(self.lora_A(self.lora_dropout(x))).to(expected_dtype) * self.scaling
                result += output
            else:
                output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
                result += output
            # print("Lora A weights: ", self.lora_A.weight[0])
            # print("Lora B weights: ", self.lora_B.weight[0])
        return result


##* Quip Implementation *##
class FusedLinearQuantLtQuip(FusedQuantizedLinear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            fuse_dim, 
            fuse_sizes, 
            in_features,
            out_features,
            codesz,
            packsz,
            pack_out,
            idx_dtype,
            codebook_version,
            lora_rank,
            rescale_WH,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            is_cuda=True,
            **kwargs,
    ):
        ## TODO Initialize FusedQuantizedLinear
        FusedQuantizedLinear.__init__(self, 
                                fuse_dim=fuse_dim, 
                                fuse_sizes=fuse_sizes, 
                                in_features=in_features,
                                out_features=out_features,
                                codesz=codesz,
                                packsz=packsz,
                                pack_out=pack_out,
                                idx_dtype=idx_dtype,
                                codebook_version=codebook_version,
                                outlier_channel_split=False,
                                rank=lora_rank,
                                rescale_WH=rescale_WH)
        LoraLayer.__init__(
            self, 
            r=r, 
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout, 
            merge_weights=False
        )
        # Actual trainable parameters
        if r > 0:
            # self.lora_A = nn.Linear(in_features, r, bias=False)
            # self.lora_B = nn.Linear(r, out_features, bias=False)
            self.scaling = self.lora_alpha / self.r #? This could be in a ModuleDict, but we now assume this is shared across all adapters ?#

            self.lora_dropout = nn.ModuleDict({})
            self.lora_A = nn.ModuleDict({})
            self.lora_B = nn.ModuleDict({})
            for i in range(len(self.fuse_sizes)):
                lora_dropout_layer = nn.Dropout(p=lora_dropout)
                self.lora_dropout.update(nn.ModuleDict({str(i): lora_dropout_layer}))
                self.lora_A[str(i)] = nn.Linear(self.fuse_sizes[i], r, bias=False)
                self.lora_B[str(i)] = nn.Linear(r, self.fuse_sizes[i], bias=False)

            # Freezing the tensors except for LoRA parameters
            self.Qidxs.requires_grad = False
            self.SU.requires_grad = False
            self.SV.requires_grad = False
            self.Wscale.requires_grad = False
            if self.A is not None: self.A.requires_grad = False
            if self.B is not None: self.B.requires_grad = False
            if self.scaleWH is not None: self.scaleWH.requires_grad = False

            #* Fused *#
            self.fuse_scales.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, "lora_A"):
            for i in range(len(self.fuse_sizes)):
                # initialize A the same way as the default for nn.Linear and B to zero
                nn.init.kaiming_uniform_(self.lora_A[str(i)].weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B[str(i)].weight)
            # # initialize A the same way as the default for nn.Linear and B to zero
            # nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            # nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor):
        # breakpoint()
        fuse_result_tuple = super().forward(x) #? LoRA is only applied to attention module? 
        query_states, key_states, value_states = fuse_result_tuple

        if self.disable_adapters:
            return fuse_result_tuple
        elif self.r > 0:
            if not torch.is_autocast_enabled():
                expected_dtype = query_states.dtype
                
                if x.dtype != torch.float32:
                    x = x.float()
                
                for i in range(len(self.fuse_sizes)):
                    query_states += self.lora_B[str(i)](self.lora_A[str(i)](self.lora_dropout[str(i)](x))).to(expected_dtype) * self.scaling
                    key_states += self.lora_B[str(i)](self.lora_A[str(i)](self.lora_dropout[str(i)](x))).to(expected_dtype) * self.scaling
                    value_states += self.lora_B[str(i)](self.lora_A[str(i)](self.lora_dropout[str(i)](x))).to(expected_dtype) * self.scaling
                #output = self.lora_B(self.lora_A(self.lora_dropout(x))).to(expected_dtype) * self.scaling
                #result += output
            else:
                for i in range(len(self.fuse_sizes)):
                    query_states += self.lora_B[str(i)](self.lora_A[str(i)](self.lora_dropout[str(i)](x))) * self.scaling
                    key_states += self.lora_B[str(i)](self.lora_A[str(i)](self.lora_dropout[str(i)](x))) * self.scaling
                    value_states += self.lora_B[str(i)](self.lora_A[str(i)](self.lora_dropout[str(i)](x))) * self.scaling
                # output = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
                # result += output
            # print("Lora A weights: ", self.lora_A.weight[0])
            # print("Lora B weights: ", self.lora_B.weight[0])
        return fuse_result_tuple #TODO: Check this: seems working. 
    

def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError
