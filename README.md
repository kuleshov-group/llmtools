# LLMTools: Run & Finetune LLMs on Consumer GPUs

LLMTools is a user-friendly library for running and finetuning LLMs in low-resource settings. Features include:
* 🔨 LLM finetuning in 2-bit, 3-bit, 4-bit precision using the ModuLoRA algorithm
* 🐍 Easy-to-use Python API for quantization, inference, and finetuning
* 🤖 Modular support for multiple LLMs, quantizers, and optimization algorithms
* 🤗 Share all your finetuned LLMs on the HuggingFace Hub

LLMTools is a research project at Cornell University, and is based on the following publications.

> - Junjie Yin, Jiahao Dong, Yingheng Wang, Christopher De Sa, Volodymyr Kuleshov ModuLoRA: Finetuning 2-Bit LLMs on Consumer GPUs by Integrating with Modular Quantizers. TMLR 2023, **Featured Certificate**. [LINK](https://arxiv.org/pdf/2309.16119)
> - Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa. QuIP: 2-Bit Quantization of Large Language Models with Guarantees. NeurIPS 2023, **Spotlight**. [LINK](https://arxiv.org/abs/2307.13304)
> - Tseng, Albert, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa. "Quip#: Even better LLM quantization with hadamard incoherence and lattice codebooks." arXiv preprint arXiv:2402.04396 (2024). [LINK](https://arxiv.org/html/2402.04396v1)

LLMTools implements low precision LoRA, a new memory-efficient finetuning algorithm that integrates with an *arbitrary* quantization module. When using the state-of-the-art QUIP# quantizer, ModuLoRA can finetune 2-bit LLMs for the first time (see [results](#benchmark) below).
 
For a detailed walk through of LLMTools and ModuLoRA, please refer to our [**Blog Post**](https://oseyincs.io/llmtools/). 

Previous release of LLMTools can be found [here](https://github.com/kuleshov-group/llmtools/tree/llmtools-optq), where we integrate OPTQ as our quantization module. We will integrate two release shortly.  

_Please open a GitHub ticket if you have any questions about the code or ModuLoRA in general._

## Overview

LLMTools provides an interface similar to the HuggingFace library for loading, generating, and finetuning quantized LLMs.

```python
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
    )

# decode and print
output = tokenizer.decode([el.item() for el in generated_ids[0]])
print(output)
```

LLMTools comes with a patched version of the PEFT library that can be used to finetune the quantized models using the ModuLoRA method.

```python
from transformers import AutoTokenizer
from llmtools.llms.autollm import AutoLLMForCausalLM
from llmtools.engine.lora.peft import quant_peft

# load model and tokenizer
model_name = 'relaxml/Llama-1-65b-E8P-2Bit' # pulls from HF hub
llm, quip_config = AutoLLMForCausalLM.from_pretrained(model_name, load_in_quip=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

# set up finetuning config
from llmtools.engine.lora.config import FinetuneConfig
tune_config = FinetuneConfig(
    # ... set up finetuning config
)

# set up lora    
lora_config = quant_peft.LoraConfig(
    # ... create a lora config object
)

model = quant_peft.get_peft_model(llm, lora_config)
    

# load stanford alpaca data
data = # ... load the data

# training args
import transformers
training_arguments = transformers.TrainingArguments(
    # ... set up batch size, etc., in the usual way
)

# start trainer
from llmtools.engine.hf.trainer import Trainer
trainer = Trainer(
    model=model,
    train_dataset=data.train_data,
    eval_dataset=data.val_data,
    args=training_arguments,
    data_collator=transformers.DataCollatorForLanguageModeling(
        tokenizer, mlm=False
    ),
)

# start training
trainer.train()

# Save Model
model.save_pretrained(tune_config.lora_out_dir)
```

For more examples of how to perform model quantization, inference, and finetuning take a look at the `examples` folder.

## Installation

### Requirements

LLMTools requires a UNIX environment supporting Python (3.9) and PyTorch (we tested with 2.1.1 with 12.1 Cuda). See `requirements.txt` for details.

First clone the repo and its submodules

```python
git clone --recursive https://github.com/kuleshov-group/llmtools.git
git submodule update --init --recursive
```

To ensure maximum reproducibility, consider creating a new conda environment:
```python
conda create -n llmtools python=3.9.18
conda activate llmtools
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
cd llmtools/third-party/quip
python -m pip install -r requirements.txt
cd ../../
python -m pip install -r requirements.txt
```
LLMTools also requries an NVIDIA GPU (Pascal architecture or newer); other platforms are currently unsupported.

### Setup

We use `distutils` to package llmtools. If you are not running conda, you can also create a `virtualenv`.
``` 
cd llmtools/third-party/quip/quiptools
python setup.py install           # installs quantizaiton module in your environment
cd ../../../
python setup.py install           # installs llmtools in your environment
```

Note that this process compiles and installs a custom CUDA kernel that is necessary to run quantized models.

## Running LLMTools

### Download Quantized Models


First, start by downloading the quantized model weights. Currently the LLAMA 1/2 models are supported with 2-bit and 4-bit precisions. (Pre-quantized weights with Quip# and OPTQ can be found [here](#quantized-model-weights))

```python
from llmtools.llms.autollm import AutoLLMForCausalLM
model_name = 'relaxml/Llama-1-30b-E8P-2Bit' # pulls from HF hub
llm, quip_config = AutoLLMForCausalLM.from_pretrained(model_name, load_in_quip=True, device_map="auto")
```


### Generation

Next, we generate text fron a quantized model. 

```python
import torch
from transformers import AutoTokenizer
from llmtools.llms.autollm import AutoLLMForCausalLM

# load model and tokenizer
model_name = 'relaxml/Llama-1-65b-E8P-2Bit' # pulls from HF hub
llm, quip_config = AutoLLMForCausalLM.from_pretrained(model_name, load_in_quip=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
```

We encode the prompt and generate.
```python
# encode prompt
prompt = 'The pyramids were built by'
input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')

# generate text
with torch.no_grad():
    generated_ids = llm.generate(
        inputs=input_ids,
        do_sample=True,
    )

# decode and print
output = tokenizer.decode([el.item() for el in generated_ids[0]])
print(output)
```

### Finetune A Base Model

Lastly, we can finetune quantized models. We again start by loading a model.

```python
from transformers import AutoTokenizer
from llmtools.llms.autollm import AutoLLMForCausalLM

# load model and tokenizer
model_name = 'relaxml/Llama-1-65b-E8P-2Bit' # pulls from HF hub
llm, quip_config = AutoLLMForCausalLM.from_pretrained(model_name, load_in_quip=True, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
```

We can set parameters via the finetune config object.
```python
# set up finetuning config
from llmtools.engine.lora.config import FinetuneConfig
tune_config = FinetuneConfig(
    dataset=None, 
    data_type = 'alpaca',
    lora_out_dir='./llama-65b-quantized-lora', 
    mbatch_size=1,
    batch_size=2,
    epochs=3,
    lr=2e-4,
    cutoff_len=256,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    val_set_size=0.2,
    warmup_steps=50,
    save_steps=50,
    save_total_limit=3,
    logging_steps=10,
)
```

We instatiate a PEFT model using our custom patched version of PEFT.

```python
# set up lora    
from llmtools.engine.lora.peft import quant_peft
lora_config = quant_peft.LoraConfig(
    task_type="CAUSAL_LM",
    r=tune_config.lora_r,
    lora_alpha=tune_config.lora_alpha,
    lora_dropout=tune_config.lora_dropout,
    bias="none",
    target_modules=["qkv_proj"],
)

# create a new lora from config
model = quant_peft.get_peft_model(llm, lora_config)
```

Next, we load the finetuning data. The library has helpers to pre-load common datasets like Alpaca.
```python
# load stanford alpaca data
from llmtools.data import TrainSAD
data = TrainSAD(
    tune_config.dataset, 
    tune_config.val_set_size, 
    tokenizer, 
    tune_config.cutoff_len
)
data.prepare_data() # this tokenizes the dataset
```

The training process is identical to that of standard LoRA and PEFT.
```python
# training args
import transformers
training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=tune_config.mbatch_size,
    gradient_accumulation_steps=tune_config.gradient_accumulation_steps,
    warmup_steps=tune_config.warmup_steps,
    num_train_epochs=tune_config.epochs,
    learning_rate=tune_config.lr,
    fp16=True,
    logging_steps=tune_config.logging_steps,
    evaluation_strategy="no",
    save_strategy="steps",
    eval_steps=None,
    save_steps=tune_config.save_steps,
    output_dir=tune_config.lora_out_dir,
    save_total_limit=tune_config.save_total_limit,
    load_best_model_at_end=False,
    ddp_find_unused_parameters=False if tune_config.ddp else None,
)

# start trainer
from llmtools.engine.hf.trainer import Trainer
trainer = Trainer(
    model=model,
    train_dataset=data.train_data,
    eval_dataset=data.val_data,
    args=training_arguments,
    data_collator=transformers.DataCollatorForLanguageModeling(
        tokenizer, mlm=False
    ),
)

# start training
trainer.train()
```

Finally, we can save and load LoRA adapters in the usual way.
```python
# Save Model
model.save_pretrained(tune_config.lora_out_dir)

```

## Quantized Model Weights

Quantized QuIP# models are available on the official [QuIP# codebase](https://github.com/Cornell-RelaxML/quip-sharp) and on [HF Hub](https://huggingface.co/relaxml). 


In our [earlier release](https://github.com/oseyosey/llmtools/tree/03f06f396df0ab3bd7ef9e6e4f8a666795f4abab), we release our quantized OPTQ weights for LLAMA model set on HF hub for easy access. (We will integrate two versions of the codebase shortly)

| ModuLoRA LLAMA Weights     | 4-bit              | 3-bit             |
|----------------------------------|-----------------|-----------------|
|    7B         | [🤗 Link](https://huggingface.co/kuleshov/llama-7b-4bit) | [🤗 Link](https://huggingface.co/kuleshov/llama-7b-3bit)  |
| 13B         | [🤗 Link](https://huggingface.co/kuleshov/llama-13b-4bit) |   [🤗 Link](https://huggingface.co/kuleshov/llama-13b-3bit) |
| 30B | [🤗 Link](https://huggingface.co/kuleshov/llama-30b-4bit) | [🤗 Link](https://huggingface.co/kuleshov/llama-30b-3bit) | 
|  65B| [🤗 Link](https://huggingface.co/kuleshov/llama-65b-4bit)| [🤗 Link](https://huggingface.co/kuleshov/llama-65b-3bit) |



| ModuLoRA OPT Weights     | 4-bit              | 3-bit             |
|----------------------------------|-----------------|-----------------|
|    7B         | [🤗 Link](https://huggingface.co/kuleshov/opt-6.7b-4bit) | [🤗 Link](https://huggingface.co/kuleshov/llama-6.7b-3bit)  |
| 13B         | [🤗 Link](https://huggingface.co/kuleshov/opt-13b-4bit) |   [🤗 Link](https://huggingface.co/kuleshov/opt-13b-3bit) |
| 30B | [🤗 Link](https://huggingface.co/kuleshov/opt-30b-4bit) | [🤗 Link](https://huggingface.co/kuleshov/opt-30b-3bit) | 

## Advanced Usage 

You can also finetune the quantized models with as many GPUs as you want. We provide two ways of parallelism to scale up your training. 

### Enabling NPP Training
The LLMTools library supports naive pipeline parallelism (NPP) training for our incorporated quantized models. NPP is a straightforward method for distributing a model across multiple GPUs. By loading both the model and its adapters onto several GPUs, NPP enables the basic communication of activations and gradients across GPUs. This approach essentially evenly fits the model across all available GPUs.

**How To Use NPP**

Check out this example on how to launch NPP training. 

You need to set up your device map such that the process will dispatch model's module correctly on multiple GPUs. 

```python
num_of_gpus = torch.cuda.device_count()
if num_of_gpus > 1:
    print("Enabling Naive Pipeline Parallel")
    max_memory = get_balanced_memory(
        model,
        max_memory=None,
        no_split_module_classes=["LlamaDecoderLayer", "LlamaMLP"],
        dtype='float16',
        low_zero=False,
    )

    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["LlamaDecoderLayer", "LlamaMLP"],
        dtype='float16'
    )

    model = dispatch_model(model, device_map=device_map)

```
### Enabling DDP Training
The LLMTools library also supportsData Distributed Parallel (DDP) Training. DDP duplicates the model from GPU 0 to all other GPUs. For every batch, each GPU processes its own mini-batch of data independently. During the backward pass, after local gradients have been calculated, they are averaged across all participating processes, facilitating efficient parallel processing and synchronization among the GPUs.

Note that DDP should work **if and only if** the training setup (meaning model weights, gradients + intermediate hidden states) can entirely fit a single GPU. 

**How To Use DDP**

Check out this example on how to launch DDP training. 

You need to set up device_map such that each working process will load the entire model on the correct GPU. You can set up the device_map as followed:

```python
device_index = Accelerator().process_index
device_map = {"": device_index}
```

 If used, gradient accumulation step should be evently split on multiple GPUs:

```python
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    num_of_gpus = torch.cuda.device_count()
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = tune_config.batch_size // (tune_config.mbatch_size*num_of_gpus)
    print("gradient_accumulation_steps: ", gradient_accumulation_steps)
```

You can launch ModuLoRA Finetuning with DDP training by using:

```
accelerate launch {script_name.py} --arg1 --arg2 ...
```

For more information on how to use accelerate, please see the official [accelerate doc](https://huggingface.co/docs/accelerate/en/basic_tutorials/launch) in HuggingFace.


## Benchmark


Our method shows competitive performance comparable or superior to baselines and 4bit / 8bit Bits&Bytes finetuning by [Dettmers et al., 2023](https://arxiv.org/pdf/2305.14314) on SAMSum benchmark with the [Llama (Touvron et al., 2023)](https://arxiv.org/pdf/2302.13971) model set. 4-bit 65B LLAMA models finetuned with ModuLoRA outperform the GPT-3 [LoRA baseline (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) and even reach new state-of-the-art performance on this dataset.2-bit ModuLoRA models match the performance of 8-bit LoRAs in BitsAndBytes and LLM.int8() and 4-bit LoRAs in BitsAndBytes and QLoRA.

We release complementary codebase [ModuLoRA-Experiment](https://github.com/kuleshov-group/MODULoRA-Experiment) to reproduce our results. 

| Models  | Finetuning Adaptation | # Trainable Parameters | SAMSum (Rouge 1/2/L)       |
|---------|-----------------------|------------------------|----------------------------|
| **GPT-3**   | Full Finetuning       | 175,255.8M             | 52.0 / 28.0 / 44.5         |
| **GPT-3**   | Adapter               | 40.1M                  | 53.2 / 29.0 / 45.1         |
| **GPT-3**   | LoRA                  | 4.7M                   | 53.8 / 29.8 / 45.9         |
| **Pegasus**| SliC                  | 2B                     | **54.4 / 29.9 / 45.9**     |


| LLAMA Finetuning | Quantizer | 7B | 13B | 30B | 65B |
|------------------|-----------|------------|------------|------------|------------|
| LLMA Tools (2-bit) | QuIP# (E8) | 51.3/27.3/43.7 | 52.3/29.0/45.0 | 53.3/30.2/46.0 | 54.0/30.6/46.2 |
| LLMA Tools (3-bit) | OPTQ | 51.2/28.2/44.0 | 52.4/29.6/45.1 | 53.6/30.8/46.3 | 54.1/30.9/46.5 |
| LLMA Tools (4-bit) | OPTQ | 51.7/28.3/44.4 | 53.2/30.2/46.1 | 53.9/31.2/46.9 | **54.8/31.3/47.2** |
| Bits&Bytes (4-bit) | QLoRA | 51.6/28.3/44.5 | 51.3/28.1/44.1 | 53.0/30.2/45.7 | 53.8/30.5/45.9 |
| Bits&Bytes (8-bit) | LLMA.int8() | 51.9/28.1/44.5 | 51.3/28.2/43.6 | 50.8/28.4/44.1 | 53.9/30.4/46.3 |


 

## Hardware Requirements

The following hardware is needed to run different models in LLMTools:

| Model Size | GPU Memory Requirements | Compatible GPUs |
| ----- | -------------------- | --------------- |
| 7b-2bit | 3GB | GTX 1080, RTX 2060, 3050, 3060 |
| 13b-2bit | 5GB | GTX 1080, RTX 2060, 3060, 3080 |
| 30b-2bit | 12GB |  RTX 3080, 3090, 4090, V100 |
| 65b-2bit | 21GB | A100, A40, RTX 3090, 4090, A6000, 5000 |

| Model Size | GPU Memory Requirements | Compatible GPUs |
| ----- | -------------------- | --------------- |
| 7b-4bit | 6GB | RTX 2060, 3050, 3060 |
| 13b-4bit | 10GB | GTX 1080, RTX 2060, 3060, 3080 |
| 30b-4bit | 20GB |  RTX 3080, A5000, 3090, 4090, V100 |
| 65b-4bit | 40GB | A100, 2x3090, 2x4090, A40, A6000 |

Only NVIDIA GPUs with the Pascal architecture or newer can run the current system.

## Examples

This is LLMTools running an instruction finetuned LLAMA-65B model on one NVidia A6000:

```
$ llmtools generate --model llama-65b-4bit --weights llama65b-4bit.pt --adapter alpaca-lora-65b-4bit --prompt "Write a well-thought out abstract for a machine learning paper that proves that 42 is the optimal seed for training neural networks."

The goal of this paper is to prove that 42 is the optimal seed for 
training neural networks. To do so, a set of experiments was conducted 
with various seeds ranging from 0 to 100. For each experiment, the 
neural networks were trained on a standard benchmark dataset and 
evaluated for accuracy, speed of training, and generalization accuracy. 
The results were then collected and analyzed. The analysis revealed 
that 42 consistently yielded the highest accuracy with the lowest 
generalization error, as well as the fastest training times. 
Furthermore, these results were consistent across multiple datasets 
and neural network architectures. Based on this evidence, it can be 
concluded that 42 is indeed the optimal seed for training neural 
networks. This paper also discusses the implications of this finding 
and its potential applications in the field of machine learning.

In summary, this research provides concrete evidence to support the use
of 42 as the optimal seed for training neural networks, and provides 
further insights into the optimal parameters for training neural networks 
in general. The findings of this research may have significant implications 
for the field of machine learning and the development of optimized training 
strategies for neural networks.

References
[1] X. Zhang, E. Rashid, and T. Yang, “An analysis of the optimal seed for training neural networks,” Machine Learning Journal, vol. 13, no. 1, pp. 21-34, 2022.
[2] C. Kim, T. Smith, and A. Vishwanathan, “A survey of optimization strategies for training neural networks,” Machine Learning Journal, vol. 8, no. 4, pp. 101-115, 2020.
[3] A. Krizhevsky, I. Sutskever, and G. H. Bradshaw, “Imagenet classification with deep convolutional neural networks,” J. Comput. Vis., vol. 5, no. 3, pp. 219–225, 2012.
```

In this example, the LLM produces a recipe for blueberry lasagna:
```
$ llmtools generate --model llama-65b-4bit --weights llama-65b-4bit.pt --adapter alpaca-lora-65b-4bit-e3 --instruction "Write a well-thought out recipe for a new blueberry lasagna dish." --max-length 500
Ingredients:
* 1 lb lasagna noodles
* 1/2 cup ricotta cheese
* 2 eggs
* 4 tablespoons parmesan cheese
* 2 cups blueberries
* 2 tablespoons sugar
* 2 tablespoons flour
* 2 tablespoons butter

Instructions:
1. Preheat oven to 350 degrees F.
2. Bring a large pot of salted water to a boil. Add the lasagna noodles and cook according to package directions.
3. In a medium bowl, combine the ricotta cheese, eggs, parmesan cheese, and 1 tablespoon of the flour; mix until combined.
4. In a skillet over medium-high heat, melt the butter and add the blueberries and sugar. Cook for 5 minutes, stirring occasionally.
5. Spread a layer of sauce in the bottom of a 9x13 inch baking dish. Layer with a single layer of lasagna noodles. Spread the ricotta mixture over the noodles and then layer with the blueberry mixture and another layer of noodles. Top with remaining parmesan cheese.
6. Bake for 25 minutes until cheese is melted and bubbly. Enjoy!

Note: You can make the ricotta mixture ahead of time and store it in the fridge. You can also cook the blueberry mixture ahead of time and store it in the fridge. Both should be reheated before assembling the lasagna.

Enjoy!
```

The 30B and 65B parameter models can do zero-shot chain-of-thought reasoning (i.e., "let's think step-by-step"):
```
$ llmtools generate --model llama-65b-4bit --weights /share/kuleshov/vk379/llama-65b-4bit.pt --prompt "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now? A: Let's think step-by-step."
Loading LLAMA model
Done
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. 
How many tennis balls does he have now? A: Let's think step-by-step.
Roger has 5 balls
Roger bought 2 cans
Each can has 3 balls
So, Roger has 5 + 2 x 3 = 11 balls now!
```

## Todos

This is experimental work in progress. Work that stills needs to be done:
* Out-of-the-box support for additional LLMs
* Integration with additional quantizers

## Acknowledgements

LLMTools is based on the following projects:
* The Quip and Quip# algorithm and codebase by the [Relax-ML Lab](https://relax-ml.cs.cornell.edu/)
* The PEFT repo and its implementation of LoRA
* The LLAMA, OPT, and BLOOM models by META FAIR and the BigScience consortium

## Citations

Please cite this repository if you use our code.

```
@article{yin2023modulora,
  title={Modulora: Finetuning 2-bit llms on consumer gpus by integrating with modular quantizers},
  author={Yin, Junjie and Dong, Jiahao and Wang, Yingheng and De Sa, Christopher and Kuleshov, Volodymyr},
  journal={TMLR},
  year={2024}
}
```

We also recommend you cite the above projects on which this work is based.

## Feedback

Please send feedback to [Junjie Oscar Yin](https://oseyincs.io/) and [Volodymyr Kuleshov](https://twitter.com/volokuleshov).
