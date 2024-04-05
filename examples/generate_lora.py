import torch
from transformers import AutoTokenizer
from llmtools.llms.autollm import AutoLLMForCausalLM
from llmtools.engine.lora.peft import quant_peft


def generate_quip(llm, tokenizer, sample_question, max_length=128):
    sample_question_tokens = tokenizer(sample_question, return_tensors='pt')
    sample_question_tokens = sample_question_tokens['input_ids'].to(llm.device)

    sample_question_results_tokens = llm.generate(input_ids=sample_question_tokens, do_sample=True, max_length=max_length)

    sample_question_results = tokenizer.decode(sample_question_results_tokens.squeeze(), skip_special_tokens=True)
    print("Generate Output: \n", sample_question_results)
    return sample_question_results



# model config
model_name = 'relaxml/Llama-1-7b-E8P-2Bit' # HF dir.
# model_name = 'relaxml/Llama-1-7b-E8PRVQ-4Bit' # HF dir-4bit

device_map = "auto"
# load model
llm, quip_config = AutoLLMForCausalLM.from_pretrained(model_name, load_in_quip=True, device_map=device_map)
llm.eval()


# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device_map, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

llm.eval()


#load lora adapter from existing checkpoint
adapter_path = '' # can generate this via finetune.py

model = quant_peft.PeftModel.from_pretrained(
    llm, 
    adapter_path, 
    device_map='auto'
)

print(f'Adapter Loaded: {adapter_path}')


##* Random Test Example *##

sample_question2 = "Write a short story about rockets:"

generate_quip(model, tokenizer, sample_question2, max_length=128)
