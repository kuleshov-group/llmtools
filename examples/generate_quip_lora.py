import torch
from transformers import AutoTokenizer, GenerationConfig
from llmtools.llms.autollm import AutoLLMForCausalLM
from llmtools.utils import to_half_precision
from llmtools.engine.lora.peft import quant_peft
from quip.lib.utils import LMEvalAdaptor

from utils import *


def generate_quip(lm_eval_model, llm, tokenizer, sample_question, max_length=128, eos_token_id=2):
    sample_question_tokens = tokenizer(sample_question, return_tensors='pt') # sample_question_tokens = lm_eval_model.tok_encode(sample_question)
    sample_question_tokens = sample_question_tokens['input_ids'].to(llm.device)
    print(sample_question_tokens)

    #sample_question_results_tokens = lm_eval_model._model_generate(sample_question_tokens, max_length=max_length, eos_token_id=eos_token_id)
    sample_question_results_tokens = llm.generate(input_ids=sample_question_tokens, do_sample=False, max_length=max_length)
    #outputs = model.generate(input_ids=input_ids, do_sample=True, max_new_tokens = 45)

    # the model not stopping does not mean that it is not adding the eos_token but rather not predicting it.
    sample_question_results = tokenizer.decode(sample_question_results_tokens.squeeze())
    print("Output: ", sample_question_results)
    return sample_question_results


#*  model config: LLAMA-2 Model Set (will NOT produce tokenizer warning) *#
#model_name = '/share/kuleshov/jy928/llmtools-2bit/quip/quantized_weights/llama2-quip-7b'

#*  model config: LLAMA-1 Model Set (will produce tokenizer warning) *#
model_name = '/share/kuleshov/jy928/llmtools-2bit/quip/quantized_weights/llama1-quip-7b-D4'
#model_name = '/share/kuleshov/jy928/llmtools-2bit/quip/quantized_weights/llama1-quip-13B-D4'

# load model, tokenizer, and quip_config
llm, tokenizer, quip_config = AutoLLMForCausalLM.from_pretrained(model_name)
llm.eval()


##? Defining our own tokenizer. ?##
# tokenizer_name = "relaxml/Llama-1-7b-hf"
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)


#* Fixing Tokenizer and subsquent model *#
tokenizer = fix_tokenizer(tokenizer)
llm = fix_model(llm, tokenizer, use_resize=False)


#load lora adapter from existing checkpoint
adapter_path = '/share/kuleshov/jy928/llmtools-2bit/experiment/llama1_adapters/llama1-samsum-2bit-7b-fix-seed42-mb2' # can generate this via finetune.py
model = quant_peft.PeftModel.from_pretrained(
    llm, adapter_path, 
    device_map='auto'
)

print(f'loaded adapter: {adapter_path}')

### QUIP Specific Packages ###
lm_eval_model = LMEvalAdaptor(quip_config["_name_or_path"], model, tokenizer, batch_size=1)
print("QUIP Tokenizer Model: ", quip_config["_name_or_path"])


##* Random Test Example *##
sample_question1 = "Write a well-thought out recipe for a new blueberry lasagna dish: "
sample_question2 = "Write a short story about rockets:"


##* SAMSum Train Example *##
sample = "Amanda: I baked cookies. Do you want some? Jerry: Sure! Amanda: I'll bring you tomorrow :-)"
sample_question_samsum_train = f"### Summarize this: {sample}\n ### Output: "

# ##* SAMSum Test Example *##
# sample = "Hannah: Hey, do you have Betty's number? Amanda: Lemme check Hannah: <file_gif> Amanda: Sorry, can't find it. Amanda: Ask Larry Amanda: He called her last time we were at the park together Hannah: I don't know him well Hannah: <file_gif> Amanda: Don't be shy, he's very nice Hannah: If you say so.. Hannah: I'd rather you texted him Amanda: Just text him 🙂 Hannah: Urgh.. Alright Hannah: Bye Amanda: Bye bye"
# sample_question_samsum_test = f"### Summarize this: {sample}\n ### Output: "


generate_quip(lm_eval_model, model, tokenizer, sample_question1, max_length=128, eos_token_id=2)

generate_quip(lm_eval_model, model, tokenizer, sample_question2, max_length=128, eos_token_id=2)

generate_quip(lm_eval_model, model, tokenizer, sample_question_samsum_train, max_length=128, eos_token_id=2)