# from llmtools.llms.config import AutoQuantConfig, LLMType

OPT_MODELS  = [
    "opt-6.7b-4bit", "opt-13b-4bit",
    "opt-6.7b-3bit", "opt-13b-3bit",
]

def get_opt_config(model):
    if '4bit' in model:
        bits = 4
    elif '3bit' in model:
        bits = 3
    elif '2bit' in model:
        bits = 2

    if '6.7b' in model:
        hf_config_name = "facebook/opt-6.7b"
    elif '13b' in model:
        hf_config_name = "facebook/opt-13b"

    raise NotImplementedError()

    llm_config = AutoQuantConfig(
        name=model,
        model_type=LLMType.OPT,
        hf_config_name=hf_config_name,
        hf_tokenizer_config="",
        bits=bits
    )
    return llm_config
