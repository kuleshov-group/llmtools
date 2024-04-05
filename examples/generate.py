import torch
from transformers import AutoTokenizer
from llmtools.llms.autollm import AutoLLMForCausalLM

# load quip model and tokenizer
model_name = 'relaxml/Llama-1-7b-E8P-2Bit' # pulls from HF hub
llm, quip_config = AutoLLMForCausalLM.from_pretrained(model_name, load_in_quip=True, device_map="auto")
llm.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token


# encode prompt
prompt = 'The pyramids were built by'
input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')

# generate text
with torch.no_grad():
    generated_ids = llm.generate(
        inputs=input_ids,
        do_sample=True,
        max_new_tokens=128,
    )

# decode and print
output = tokenizer.decode([el.item() for el in generated_ids[0]])
print(output)