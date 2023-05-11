import torch
from typing import Dict, Any
from datasets import load_dataset
from llmtune.engine.data.abstract import AbstractTrainData

# GPT4All-like Data
class TrainGPT4All(AbstractTrainData):
    def __init__(self, dataset: str, val_set_size: int, tokenizer, cutoff_len) -> None:
        super().__init__(dataset, val_set_size, tokenizer, cutoff_len)

    def tokenize(self, prompt: str, use_eos_token=True, **kwargs) -> Dict[str, Any]:
        pass

    def tokenize_inputs(self, examples):
        max_length = self.cutoff_len
        input_ids = torch.full((len(examples["prompt"]), max_length), self.tokenizer.pad_token_id)
        # ignore bos
        newline_tokens = self.tokenizer("\n", return_tensors="pt")["input_ids"][0, 1:]

        out = {"labels": [], "attention_mask": []}
        for i, (prompt, response) in enumerate(zip(examples["prompt"], examples["response"])):
            input_tokens = self.tokenizer(prompt, truncation=True, max_length=max_length // 2, return_tensors="pt")["input_ids"].squeeze()
            if input_tokens.dim() == 0:
                input_tokens = input_tokens.unsqueeze(0)

            input_len = len(input_tokens)

            # plus one since we remove bos from response
            # but we subtract one since we want to add eos token
            remaining_tokens = max_length - input_len - len(newline_tokens) + 1
            # remove bos
            target_tokens = self.tokenizer(response, truncation=True, max_length=remaining_tokens, return_tensors="pt")["input_ids"].squeeze()[1:]

            input_ids[i, :input_len] = input_tokens
            # add newline between prompt and response
            newline_plus_inputs = input_len + len(newline_tokens)
            input_ids[i, input_len: newline_plus_inputs] = newline_tokens

            # add target tokens, remove bos
            input_ids[i, newline_plus_inputs: newline_plus_inputs + len(target_tokens)] = target_tokens
            # add eos token, enforce stopping if we don't truncate
            # we don't want long code to stop generating if truncated during training
            if newline_plus_inputs + len(target_tokens) < max_length:
                input_ids[i, newline_plus_inputs + len(target_tokens)] = self.tokenizer.eos_token_id

            labels = input_ids[i].clone()
            labels[: newline_plus_inputs] = -100
            labels[labels == self.tokenizer.pad_token_id] = -100
            # to debug this, can set all values == -100 to the pad token, then assert that tokenizer.decode(labels, skip_special_tokens=True).strip() == response

            attention_mask = input_ids[i].ne(self.tokenizer.pad_token_id).int()

            out["labels"].append(labels)
            out["attention_mask"].append(attention_mask)

        out["input_ids"] = input_ids

        out = {k: torch.stack(v) if isinstance(v, list) else v for k, v in out.items()}

        return out

    def prepare_data(self, **kwargs) -> None:
        dataset = load_dataset("json", data_files=self.dataset)

        self.val_data = None
        if self.val_set_size > 0:
            dataset = dataset["train"].train_test_split(
                test_size=self.val_set_size, shuffle=True, seed=42  # ! Seed = 42 (?)
            )
            train_dataset, val_dataset = dataset["train"], dataset["test"]

            # tokenize inputs and return labels and attention mask
            val_dataset = val_dataset.map(
                lambda ele: self.tokenize_inputs(ele),
                batched=True,
                remove_columns=["source", "prompt"],
            )
            self.val_data = val_dataset.with_format("torch")
        else:
            train_dataset = dataset["train"]

        train_dataset = train_dataset.map(
            lambda ele: self.tokenize_inputs(ele),
            batched=True,
            remove_columns=["source", "prompt"],
        )
        self.train_data = train_dataset.with_format("torch")
