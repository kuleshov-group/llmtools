import torch
from transformers import AutoTokenizer
from llmtune.llms.autollm import AutoLLMForCausalLM
from llmtune.utils import to_half_precision

# model config
model_name = 'kuleshov/llama-7b-4bit'
# model_name = './llama-7b-quantized' # can generate local dir via quantize.py
tokenizer_name = 'huggyllama/llama-13b'
DEV = 'cuda'

# load model
llm = AutoLLMForCausalLM.from_pretrained(model_name).to(DEV)
llm.eval()
llm = to_half_precision(llm)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# encode prompt
prompt = 'The pyramids were built by'
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEV)

# generation config
min_length=10
max_length=200
top_p=.95
top_k=25
temperature=1.0

# generate text
with torch.no_grad():
    generated_ids = llm.generate(
        inputs=input_ids,
        do_sample=True,
        min_length=min_length,
        max_length=max_length,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )

# decode and print
output = tokenizer.decode([el.item() for el in generated_ids[0]])
print(output)