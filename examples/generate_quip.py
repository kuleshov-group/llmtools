import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from llmtools.llms.autollm import AutoLLMForCausalLM
from llmtools.utils import to_half_precision

## QUIP Path
from quip.lib.utils import LMEvalAdaptor


def generate_quip(llm, tokenizer, sample_question, max_length=5):
    sample_question_tokens = tokenizer(sample_question, return_tensors='pt') # sample_question_tokens = lm_eval_model.tok_encode(sample_question)
    sample_question_tokens = sample_question_tokens['input_ids'].to(llm.device)
    # print(sample_question_tokens)

    sample_question_results_tokens = llm.generate(input_ids=sample_question_tokens, do_sample=True, top_p=0.9, max_new_tokens = 512)

    # the model not stopping does not mean that it is not adding the eos_token but rather not predicting it.
    sample_question_results = tokenizer.decode(sample_question_results_tokens[0].detach().cpu().numpy(), skip_special_tokens=True)
    print("Output: ", sample_question_results)
    return sample_question_results


#*  model config: LLAMA-2 Model Set (will NOT produce tokenizer warning) *#
#model_name = '/share/kuleshov/jy928/llmtools-2bit/quip/quantized_weights/llama2-quip-7b'

#*  model config: LLAMA-1 Model Set (will produce tokenizer warning) *#
# model_name = '/share/kuleshov/jy928/llmtools-2bit/quip/quantized_weights/llama1-quip-7b-D4'
#model_name = '/share/kuleshov/jy928/llmtools-2bit/quip/quantized_weights/llama1-quip-13B-D4'

model_name = 'relaxml/Llama-1-65b-E8P-2Bit' # HF dir.

device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1

# load model (QUIP)
llm, quip_config = AutoLLMForCausalLM.from_pretrained(model_name, "QUIP", device_map=device_map)
llm.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device_map, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

if not ddp and torch.cuda.device_count() > 1:
    llm.is_parallelizable = True
    llm.model_parallel = True


### QUIP Specific Packages ###
# lm_eval_model = LMEvalAdaptor(quip_config["_name_or_path"], llm, tokenizer, batch_size=1)
# print("QUIP Tokenizer Model: ", quip_config["_name_or_path"])


##* Random Test Example *##
# sample_question1 = "Write a well-thought out recipe for a new blueberry lasagna dish: "
# sample_question2 = "Write a short story about rockets:"


##* SAMSum Train Example *##
sample = f" Finn: Hey Zadie: Hi there! What's up? Finn: All fine. You? Zadie: Not bad, thanks Finn: Look, I was thinking of going to this neighborhood called Elephant and Castle tomorrow, it's apparently full of Latin American stuff. Fancy joining? Zadie: Sure! But what's stuff? 😂 Finn: lol So apparently it's a place were random people from Latin America (meaning fuck knows which countries) started running small businesses and restaurant, and a nice little community was formed Zadie: Oh cool Finn: Then capitalism came and it's all going to be demolished soon, so it's like the last chance to go Zadie: What a shame :( Yeah, I haven't had Latin American 😂 food for ages so I'm totally up for it Finn: Can't wait to taste this cuisine of unspecified Latino origin lol Zadie: Finn: But we can specify time and place if and only if you wish Zadie: I might be tempted to lol I'd say early evening, 2-ish? Finn: Yeah, that's fine by me. So most of the places we want to visit are in this Elephant and Castle shopping centre. Shall I see you at the main entrance, wherever that is Zadie: 2 o'clock at unspecified main entrance then? Sounds good to mw Finn: Yer Zadie: Cool, see you there! And thanks so much for remembering about me Finn: Thanks for saying yes to such an ill-defined plan lmao Zadie: Ha ha You know I love those Finn: See you tomorrow then Zadie: Yep Call me if you get lost Finn: I will I will 🤙 byeeee Zadie: Toodles"
sample_question_samsum = f"### Summarize this: {sample}\n ### Output: "

generate_quip(llm, tokenizer, sample_question_samsum, max_length=512)



