import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description='Python script to work with models')
parser.add_argument('--model_name', type=str, help='Name of the model', required=True)
parser.add_argument('--adapter', type=str, help='Path to store adapter weight', required=True)
parser.add_argument('--seed', type=int, help='model seed number', required=True)
parser.add_argument('--file_name', type=str, help='file name to store predictions and acc', required=True)
parser.add_argument('--checkpoint_name', type=str, help='folder name to store all the check points', required=True)
parser.add_argument('--start_index', type=int, help='starting index ', required=True)
parser.add_argument('--end_index', type=int, help='end index', required=True)

# Parse the arguments
args = parser.parse_args()

# Use the command line arguments in your script
print('Model Name:', args.model_name)
print('Adapter Path: ', args.adapter)
print('Seed: ', args.seed)

import random
import json
import os

#for eval
import pickle

# import wandb
import torch
import numpy as np
# import bitsandbytes as bnb
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, DataCollatorForTokenClassification, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments, logging, TrainerCallback, TrainerState, TrainerControl, BitsAndBytesConfig
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from peft import get_peft_model, LoraConfig, prepare_model_for_int8_training
from datasets import load_dataset

from utils import *
from data import *

import evaluate
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

from llmtools.executor import load_adapter
from llmtools.llms.autollm import AutoLLMForCausalLM
from llmtools.engine.lora.config import FinetuneConfig
from llmtools.data import TrainSAD
from llmtools.engine.lora.peft import quant_peft
from llmtools.utils import to_half_precision


output_dir = args.adapter
seed = args.seed

set_random_seed(seed)
logging.set_verbosity_info()

# with open(config_file, "r") as r:
#     config = json.load(r)

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1


dataset = load_dataset('samsum')
train_records = dataset['train']
val_records = dataset['test']
#random.shuffle(train_records)
print("train_record[0]: ",train_records[0])

## Config for llama 7-b
model_type = "causal"
templates_path = "llama_lora_samsum.json"
only_target_loss = False
mode = "instruct"


# model config
llmtune_model_name = args.model_name


#* load model (QUIP) *#
model, _, quip_config = AutoLLMForCausalLM.from_pretrained(llmtune_model_name)
model.eval()


##? Defining our own tokenizer: Fixed the issue with generation! ?##
tokenizer = AutoTokenizer.from_pretrained(llmtune_model_name, use_fast=False)


#* Fixing the model and tokenizer for Experiment*#
tokenizer = fix_tokenizer(tokenizer)
model = fix_model(model, tokenizer, use_resize=False)


# Default model generation params
model.config.num_beams = 5

if not ddp and torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True


lora_config = quant_peft.LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


#* Loading the adapter *#
# model = load_adapter(model, adapter_path=output_dir, lora_config=lora_config)
# model = quant_peft.PeftModel.from_pretrained(
#     model, output_dir, 
#     device_map='auto',
#     torch_dtype=torch.float32
# )


# print(output_dir, 'loaded')

# Metric
metric = evaluate.load("rouge")

def evaluate_peft_model_samsum(sample,max_target_length=45):
    # Load dataset from the hub and get a sample
    sample_word = f"### Summarize this: {sample}\n ### Output: "
    input_ids = tokenizer(sample_word, return_tensors="pt", truncation=True).input_ids.cuda()
    with torch.autocast("cuda"):
        #outputs = model.generate(input_ids=input_ids, do_sample=True, top_p=0.9, max_new_tokens = 512)
        outputs = model.generate(input_ids=input_ids, do_sample=True, top_p=0.9, max_new_tokens = 512)
    output = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True).replace(sample_word,"")
    output = output.strip()
    print(f"Output:\n{output}")
    # Some simple post-processing
    return output


def rouge_compute(predictions,references):
    rogue = metric.compute(predictions=predictions, references=references, use_stemmer=True)
    return rogue


def store_pred(file_name_pickle_pred,file_name_pickle_ref,predictions,references):
    with open(file_name_pickle_pred, "wb") as fp:   #Pickling
        pickle.dump(predictions, fp)
    with open(file_name_pickle_ref, "wb") as fp:   #Pickling
        pickle.dump(references, fp)


##Arguments setting
start_index = args.start_index
end_index = args.end_index
eval_len =  end_index - start_index
eval_save_len = eval_len // 10
print("Evaluation will start at: ", start_index)
print("Evaluation will end at: ", end_index)
print(f'Evaluation will save at every {eval_save_len} steps')


## Create Check point Folder
checkpoint_path = f'{args.checkpoint_name}_{start_index}_{end_index}'

current_directory = os.getcwd()
final_directory = os.path.join(current_directory, checkpoint_path)
if not os.path.exists(final_directory):
   os.makedirs(final_directory)



predictions = []
references_origin = val_records['summary'][start_index:end_index]
references = []

count_eval = 0


for idx in tqdm(range(start_index, end_index)):
    sample = val_records['dialogue'][idx]
        # Load dataset from the hub and get a sample
    sample_word = f"### Summarize this: {sample}\n ### Output: "
    input_ids = tokenizer(sample_word, return_tensors="pt", truncation=True).input_ids.cuda()

    print("length of input ids:", len(input_ids[0]))
    # if (len(input_ids[0]) < 300): 
    with torch.inference_mode(), torch.autocast("cuda"):
        outputs = model.generate(input_ids=input_ids, do_sample=True, top_p=0.9, max_new_tokens = 45)
        output = tokenizer.decode(outputs[0].detach().cpu().numpy(), skip_special_tokens=True).replace(sample_word,"")
    output = output.strip()
    print(f"Model Output: \n{output}")
    predictions.append(output)
    print(f"Reference Output: \n {references_origin[count_eval]}")
    references.append(references_origin[count_eval])
    count_eval+=1

    ## Detecting checkpoing
    if (count_eval%eval_save_len == 0):
       print(f'=>=>Checkpointing at {count_eval} steps<=<=')

       predictions_step = [s.strip() for s in predictions]
       print("prediction_step: ", predictions_step)
       references_step = references
       print("references_step: ", references_step)
       rouge = rouge_compute(predictions_step,references_step)
       checkpoint_name_txt = f'{final_directory}/{count_eval}.txt'
       checkpoint_name_pred = f'{final_directory}/{count_eval}_pred' ## pickle file for pred list
       checkpoint_name_ref = f'{final_directory}/{count_eval}_ref' ## pickle file for ref list
       ## writing pickle file
       store_pred(checkpoint_name_pred,checkpoint_name_ref,predictions_step,checkpoint_name_ref)
       with open(checkpoint_name_txt, "w") as f:
            for item in predictions_step:
                # write each item on a new line
                f.write("%s\n" % item)
            f.write(f'Seed: {seed}')
            f.write(f"Rogue1: {rouge['rouge1']* 100:2f}%")
            f.write(f"rouge2: {rouge['rouge2']* 100:2f}%")
            f.write(f"rougeL: {rouge['rougeL']* 100:2f}%")
            f.write(f"rougeLsum: {rouge['rougeLsum']* 100:2f}%")


predictions = [s.strip() for s in predictions]


# compute metric
rouge = metric.compute(predictions=predictions, references=references, use_stemmer=True)

file_name = args.file_name
with open(file_name, 'w') as f:
    f.write(f'Seed: {seed}')
    f.write(f"Rogue1: {rouge['rouge1']* 100:2f}%")
    f.write(f"rouge2: {rouge['rouge2']* 100:2f}%")
    f.write(f"rougeL: {rouge['rougeL']* 100:2f}%")
    f.write(f"rougeLsum: {rouge['rougeLsum']* 100:2f}%")