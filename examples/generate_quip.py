import torch
from transformers import AutoTokenizer
from llmtools.llms.autollm import AutoLLMForCausalLM
from llmtools.utils import to_half_precision


## QUIP Path
import sys
import os
parent_directory = os.path.abspath(f"{__file__}/../../")
sys.path.append(parent_directory)
from quip.lib.utils import LMEvalAdaptor


# model config
model_name = '/share/kuleshov/jy928/llmtools-2bit/quip/quantized_weights/llama2-quip-7b'
# model_name = './llama-7b-quantized' # can generate local dir via quantize.py
# tokenizer_name = 'huggyllama/llama-13b'
DEV = 'cuda'

# load model
breakpoint()
if "quip" in model_name:
    llm, tokenizer, quip_config = AutoLLMForCausalLM.from_pretrained(model_name)
    llm.eval()
else:
    # load model
    llm = AutoLLMForCausalLM.from_pretrained(model_name).to(DEV)
    llm.eval()
    llm = to_half_precision(llm)
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


### QUIP Specific Packages ###
lm_eval_model = LMEvalAdaptor(quip_config["_name_or_path"], model, tokenizer, 1)
sample_question = "Write a well-thought out recipe for a new blueberry lasagna dish: "


sample_question_tokens = tokenizer(sample_question, return_tensors='pt') # sample_question_tokens = lm_eval_model.tok_encode(sample_question)
sample_question_tokens = sample_question_tokens['input_ids'].to(lm_eval_model.device)
print(sample_question_tokens)

sample_question_results_tokens = lm_eval_model._model_generate(sample_question_tokens, max_length=128, eos_token_id=2)

# the model not stopping does not mean that it is not adding the eos_token but rather not predicting it.
sample_question_results = lm_eval_model.tok_decode(sample_question_results_tokens.squeeze())
print(sample_question_results)