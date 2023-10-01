from llmtools.llms.autollm import AutoLLMForCausalLM

# load model
model_dir = './llama-7b-quantized' # can generate this via quantize.py
llm = AutoLLMForCausalLM.from_pretrained(model_dir)

# push to hub
llm.push_to_hub(
	repo_id='kuleshov/llama-7b-4bit', 
	save_dir=model_dir,
	commit_message='first commit'
)
