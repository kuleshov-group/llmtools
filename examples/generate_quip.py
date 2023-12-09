import torch
import torch.nn as nn
from transformers import AutoTokenizer
from llmtools.llms.autollm import AutoLLMForCausalLM
from llmtools.utils import to_half_precision

## QUIP Path
from quip.lib.utils import LMEvalAdaptor


def generate_quip(lm_eval_model, llm, tokenizer, sample_question, max_length=128, eos_token_id=2):
    sample_question_tokens = tokenizer(sample_question, return_tensors='pt') # sample_question_tokens = lm_eval_model.tok_encode(sample_question)
    sample_question_tokens = sample_question_tokens['input_ids'].to(llm.device)
    print(sample_question_tokens)

    sample_question_results_tokens = lm_eval_model._model_generate(sample_question_tokens, max_length=max_length, eos_token_id=eos_token_id)

    # the model not stopping does not mean that it is not adding the eos_token but rather not predicting it.
    sample_question_results = tokenizer.decode(sample_question_results_tokens.squeeze())
    print("Output: ", sample_question_results)
    return sample_question_results


#*  model config: LLAMA-2 Model Set (will NOT produce tokenizer warning) *#
#model_name = '/share/kuleshov/jy928/llmtools-2bit/quip/quantized_weights/llama2-quip-7b'

#*  model config: LLAMA-1 Model Set (will produce tokenizer warning) *#
model_name = '/share/kuleshov/jy928/llmtools-2bit/quip/quantized_weights/llama1-quip-7b-D4'
#model_name = '/share/kuleshov/jy928/llmtools-2bit/quip/quantized_weights/llama1-quip-13B-D4'

# load model (QUIP)
llm, tokenizer, quip_config = AutoLLMForCausalLM.from_pretrained(model_name)
llm.eval()

### QUIP Specific Packages ###
lm_eval_model = LMEvalAdaptor(quip_config["_name_or_path"], llm, tokenizer, batch_size=1)
print("QUIP Tokenizer Model: ", quip_config["_name_or_path"])


##* Random Test Example *##
sample_question1 = "Write a well-thought out recipe for a new blueberry lasagna dish: "
sample_question2 = "Write a short story about rockets:"


##* SAMSum Train Example *##
sample = "Amanda: I baked cookies. Do you want some? Jerry: Sure! Amanda: I'll bring you tomorrow :-)"
sample_question_samsum = f"### Summarize this: {sample}\n ### Output: "

generate_quip(lm_eval_model, llm, tokenizer, sample_question1, max_length=128, eos_token_id=2)

generate_quip(lm_eval_model, llm, tokenizer, sample_question2, max_length=128, eos_token_id=2)

generate_quip(lm_eval_model, llm, tokenizer, sample_question_samsum, max_length=128, eos_token_id=2)



