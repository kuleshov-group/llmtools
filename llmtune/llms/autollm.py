import os
import torch
from torch import nn
from typing import Dict, List, Optional, Union
from transformers import AutoTokenizer
from transformers.utils.hub import (
    PushToHubMixin, cached_file, create_repo, 
    create_commit, CommitOperationAdd
)
from llmtune.llms.config import AutoLLMConfig, LLMType
from llmtune.llms.llama.model import load_llama, load_llama_tokenizer
from llmtune.llms.opt.model import load_opt, load_opt_tokenizer

def get_default_tokenizer(name_or_path, model_type=None):
    if model_type is not None:
        if model_type == 'llama':
            return load_llama_tokenizer(name_or_path)
        elif model_type == 'opt':
            return load_opt_tokenizer(name_or_path)
        else:
            raise ValueError()
    else:
        return AutoTokenizer.from_pretrained(name_or_path)

class AutoLLMForCausalLM(nn.Module, PushToHubMixin):
    def __init__(
        self, 
        base_model, 
        llm_config
    ):
        super().__init__()
        self.base_model = base_model
        self.llm_config = llm_config

    @property
    def is_quantized(self):
        return self.llm_config.is_quantized

    def set_quant_config(self, quant_config):
        self.llm_config.set_quant_config(quant_config)

    @property
    def device(self):
        if not self.hf_device_map:
            return self.base_model.device
        else:
            device = [
                d for d in self.hf_device_map.values() 
                if d not in {'cpu', 'disk'}
            ][0]
            return torch.device(device)

    @property
    def hf_device_map(self):
        return getattr(self.base_model, "hf_device_map", None)

    @property
    def config(self):
        return self.base_model.config

    @property
    def _keys_to_ignore_on_save(self):
        return self.base_model._keys_to_ignore_on_save

    @property
    def _no_split_modules(self):
        return self.base_model._no_split_modules
    
    def to(self, device: Union[str, torch.device]):
        self.base_model = self.base_model.to(device)
        return self

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def generate(self, **kwargs):
        with (
            torch.inference_mode(), 
            torch.amp.autocast(device_type=self.device.type)
        ):
            return self.base_model.generate(**kwargs)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.base_model.prepare_inputs_for_generation(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device_map: Optional[Union[str, Dict[str, Union[int, str]]]] = None,
        device: Optional[Union[str, int]] = None,
    ):
        # load config
        llm_config = AutoLLMConfig.from_pretrained(model_name_or_path)
        load_quantized = llm_config.quant_config is not None

        # resolve path to checkpoint (could be None)
        checkpoint = None
        if load_quantized:
            if os.path.isdir(model_name_or_path):
                checkpoint = os.path.join(
                    model_name_or_path, 'quantized_weights.pt'
                )
            else: # remote
                checkpoint = cached_file(
                    model_name_or_path, 'quantized_weights.pt'
                )
            if checkpoint is None:
                raise FileNotFoundError(
                    f"Couldn't find quantized weights in {model_name_or_path}"
                )

        # load base model
        if llm_config.model_type == LLMType.LLAMA.value:
            model = load_llama(llm_config, checkpoint)
        elif llm_config.model_type == LLMType.OPT.value:
            model = load_opt(llm_config, checkpoint)
        else:
            raise NotImplementedError(
               f'{llm_config.model_type} not supported'
            )

        return cls(model, llm_config)

    def save_pretrained(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        print('test')

        # save config
        self.llm_config.save_pretrained(save_dir)

        # save base model
        self.base_model.to('cpu')
        print(self.llm_config.quant_config)
        if not self.is_quantized:
            self.base_model.save_pretrained(save_dir)
        else:
            torch.save(
                self.base_model.state_dict(), 
                os.path.join(save_dir, 'quantized_weights.pt')
            )
            self.llm_config.base_config.model_name_or_path = save_dir

    def push_to_hub(
        self,
        repo_id: str,
        save_dir: str,
        commit_message: Optional[str] = "",
        use_auth_token: Optional[Union[bool, str]] = None,
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: Optional[bool] = False,
    ) -> str:
        
        if not os.path.exists(save_dir):
            print(f"Saving model to {save_dir}")
            self.save_pretrained(save_dir)

        repo_url = create_repo(
            repo_id=repo_id, token=token, private=private, 
            exist_ok=True, repo_type="model"
        )
        repo_id = repo_url.repo_id

        operations = [
            CommitOperationAdd(
                path_or_fileobj=os.path.join(save_dir, f), 
                path_in_repo=f
            )
            for f in os.listdir(save_dir)
        ]
        print(
            f"Uploading the following files to {repo_id}: "
            f"{','.join(os.listdir(save_dir))}"
        )
        return create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=commit_message,
            token=use_auth_token,
            create_pr=create_pr,
            repo_type="model",
        )

