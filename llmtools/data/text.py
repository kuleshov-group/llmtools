import os
from typing import Dict, Any
from datasets import Dataset
from torch.utils.data import DataLoader
from llmtools.data.abstract import AbstractTrainData

# LLaMA txt train data loader
class TrainTxt(AbstractTrainData):
    def __init__(self, dataset: str, val_set_size: int, tokenizer, cutoff_len):
        super().__init__(dataset, val_set_size, tokenizer, cutoff_len)  # TODO: Validation size isn't used
        self.cutoff_len = cutoff_len
        self.exceed_count = 0

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
            d = {
                "input_ids": result["input_ids"],
                "attention_mask": result["attention_mask"],
            }
            if (
                d["input_ids"][-1] != self.tokenizer.eos_token_id
                and len(d["input_ids"]) < self.cutoff_len
            ):
                d["input_ids"].append(self.tokenizer.eos_token_id)
                d["attention_mask"].append(1)
        else:
            result = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.cutoff_len + 1,
                padding="max_length",
            )
            d = {
                "input_ids": result["input_ids"][:-1],
                "attention_mask": result["attention_mask"][:-1],
            }
        if sum(d['attention_mask']) >= self.cutoff_len:
            self.exceed_count += 1
        return d

    @classmethod
    def format_new_rows(cls, rows, thd=128):
        r_b = ''
        new_rows = []
        for row in rows:
            if len(r_b) == 0:
                r_b += row
            else:
                r_b += '\n' + row
            if len(r_b) > thd:
                new_rows.append(r_b)
                r_b = ''
        if len(r_b) > thd:
            new_rows.append(r_b)
            r_b = ''
        return new_rows

    def prepare_data(self, thd=-1, use_eos_token=True, **kwargs):
        if os.path.isdir(self.dataset):
            rows = []
            for filename in os.listdir(self.dataset):
                with open(self.dataset + filename, 'r', encoding='utf8') as file:
                    txt = file.read()
                txt = txt.replace('\r\n', '\n').replace('\u3000', ' ')
                rows += [r for r in txt.split('\n') if r != '']
        else:
            with open(self.dataset, 'r', encoding='utf8') as file:
                txt = file.read()
            txt = txt.replace('\r\n', '\n')
            rows = [r for r in txt.split('\n') if r != '']
        if thd != -1:
            rows = self.format_new_rows(rows, thd=thd)
        data = Dataset.from_dict({"input": rows})
        data = data.shuffle().map(lambda x: self.tokenize(x["input"], use_eos_token=use_eos_token))
        print('Train Data: {:.2f}%'.format(self.exceed_count / len(data) * 100), 'outliers')
        self.train_data = data
