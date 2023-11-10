# from llmtools.llms.config import AutoQuantConfig, LLMType

LLAMA_MODELS = [
    "llama-7b-4bit", "llama-13b-4bit", "llama-30b-4bit", "llama-65b-4bit",
    "llama-7b-3bit", "llama-13b-3bit", "llama-30b-3bit", "llama-65b-3bit",
    "llama-7b-2bit", "llama-65b-2bit", 
]

def get_llama_config(model):
    if '4bit' in model:
        bits = 4
    elif '3bit' in model:
        bits = 3
    elif '2bit' in model:
        bits = 2

    if '7b' in model:
        hf_config_name = "huggyllama/llama-7b"
    elif '13b' in model:
        hf_config_name = "huggyllama/llama-13b"
    elif '30b' in model:
        hf_config_name = "huggyllama/llama-30b"
    elif '65b' in model:
        hf_config_name = "huggyllama/llama-65b"

    raise NotImplementedError()

    llm_config = AutoQuantConfig(
        name=model,
        model_type=LLMType.LLAMA,
        hf_config_name=hf_config_name,
        hf_tokenizer_config="huggyllama/llama-13b",
        bits=bits
    )
    return llm_config
