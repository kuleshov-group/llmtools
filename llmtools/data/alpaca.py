from typing import Dict, Any
from datasets import load_dataset
from llmtools.data.abstract import AbstractTrainData

DEFAULT_HF_PATH = "kuleshov/alpaca-data"

class TrainSAD(AbstractTrainData):
    def __init__(self, dataset: str, val_set_size: int, tokenizer, cutoff_len) -> None:
        super().__init__(dataset, val_set_size, tokenizer, cutoff_len)

    def tokenize(self, prompt: str, use_eos_token=True, **kwargs) -> Dict[str, Any]:
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        if use_eos_token:
            result = self.tokenizer(
                prompt + self.tokenizer.eos_token,
                truncation=True,
                max_length=self.cutoff_len,
                padding=False,
            )
            if (
                result["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(result["input_ids"]) < self.cutoff_len
            ):
                result["input_ids"].append(self.tokenizer.eos_token_id)
                result["attention_mask"].append(1)
            return result
        else:
            result = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.cutoff_len + 1,
                padding="max_length",
            )
            return {
                "input_ids": result["input_ids"][:-1],
                "attention_mask": result["attention_mask"][:-1],
            }

    def prepare_data(self, use_eos_token=True, **kwargs) -> None:
        if self.dataset:
            data = load_dataset("json", data_files=self.dataset)
        else:
            data = load_dataset(DEFAULT_HF_PATH)

        if self.val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=self.val_set_size, shuffle=True, seed=42
            )
            self.train_data = train_val["train"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
            self.val_data = train_val["test"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
        else:
            self.train_data = data["train"].shuffle().map(lambda x: self.generate_and_tokenize_prompt(x, use_eos_token=use_eos_token))
            self.val_data = None

    # Auxiliary methods
    def generate_prompt(self, data_point, **kwargs):
        return make_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"]
        )


    def generate_and_tokenize_prompt(self, data_point, **kwargs):
        prompt = self.generate_prompt(data_point, **kwargs)
        return self.tokenize(prompt, **kwargs)

def make_prompt(instruction, input_, output=""):
    return "{0}\n\n{1}\n{2}\n\n{3}\n{4}\n\n{5}\n{6}".format(
        "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
        "### Instruction:",
        instruction,
        "### Input:",
        input_,
        "### Response:",
        output
    )

def make_output(raw_output):
    return raw_output.split("### Response:")[1].strip()