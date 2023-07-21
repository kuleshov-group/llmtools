from llmtune.llms.autollm import AutoLLMForCausalLM
from llmtune.engine.quant.config import QuantConfig
from llmtune.engine.quant.gptq.executor import GPTQAlgorithm
from llmtune.data.calibration import get_calibration_loaders

# load model
model_name = 'decapoda-research/llama-7b-hf'
llm = AutoLLMForCausalLM.from_pretrained(model_name)
llm.eval()

# set up quantization config
config = QuantConfig(
	bits=4,
	dataset='c4',
	seed=0,
	nsamples=128,
	percdamp=.01,
	groupsize=64,
	act_order=True,
	nearest=False,
	save='./llama-7b-quantized'
)

# load gptq calibration data
dataloader, _ = get_calibration_loaders(
    config.dataset, 
    nsamples=config.nsamples, 
    seed=config.seed, 
    model=llm.base_model.name_or_path, 
    seqlen=llm.base_model.seqlen
)

# create quantization algorithm
gptq = GPTQAlgorithm(config)
llm = gptq.quantize(llm, dataloader)

llm.save_pretrained(config.save)
print(f'Model weights saved to: {config.save}')