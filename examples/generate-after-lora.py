import torch
from transformers import AutoTokenizer, GenerationConfig
from llmtune.llms.autollm import AutoLLMForCausalLM
from llmtune.utils import to_half_precision
from llmtune.engine.lora.peft import quant_peft

# model config
model_name = './llama-7b-quantized' # can generate this via quantize.py
tokenizer_name = 'huggyllama/llama-7b'
DEV = 'cuda'

# load model
llm = AutoLLMForCausalLM.from_pretrained(model_name).to(DEV)
llm.eval()
llm = to_half_precision(llm)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# load lora from existing checkpoint
adapter_path = './llama-7b-quantized-lora' # can generate this via finetune.py
model = quant_peft.PeftModel.from_pretrained(
    llm, adapter_path, 
    device_map='auto'
)
print(adapter_path, 'loaded')

# encode prompt
prompt = 'Write a detailed step-by-step recipe for a blueberry lasagna dish'
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEV)

# generation config
min_length=10
max_length=200
top_p=.95
top_k=25
temperature=1.0

# generate text
with torch.no_grad():
    generated_ids = model.generate(
        inputs=input_ids,
        generation_config=GenerationConfig(
	        do_sample=True,
	        min_length=min_length,
	        max_length=max_length,
	        top_p=top_p,
	        top_k=top_k,
	        temperature=temperature,
	    )
    )

# decode and print
output = tokenizer.decode([el.item() for el in generated_ids[0]])
print(output)