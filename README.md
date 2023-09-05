# LLMTools: Run & Finetune LLMs on Consumer GPUs

LLMTools is a user-friendly library for running and finetuning LLMs in low-resource settings. Features include:
* 🔨 LLM finetuning in 2-bit, 3-bit, 4-bit precision using the LP-LoRA algorithm
* 🐍 Easy-to-use Python API for quantization, inference, and finetuning
* 🤖 Modular support for multiple LLMs, quantizers, and optimization algorithms
* 🤗 Share all your LLMs on the HuggingFace Hub

LLMTools is a research project at Cornell University, and is based on the following publications.
```
Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa.  QuIP: 2-Bit Quantization of Large Language Models with Guarantees., ArXiv, https://arxiv.org/abs/2307.13304
Volodymyr Kuleshov, Oscar Yinn, Jiahao Don, Yingheng Wang, Christopher De Sa.  Low-Precision LoRA: Finetuning Large Language Models on Consumer GPUs via Modular Quantizers., ArXiv
```
LLMTools implements low precision LoRA, a new memory-efficient finetuning algorithm that integrates with an *arbitrary* quantization module. When using the state-of-the-art OPTQ quantizer, LP-LoRA can finetune 3-bit LLMs for the first time (see results below).

## Overview

LLMTools provides an interface similar to the HuggingFace library for loading, generating, and finetuning quantized LLMs.

```python
import torch
from transformers import AutoTokenizer
from llmtools.llms.autollm import AutoLLMForCausalLM

# load model and tokenizer
model_name = 'kuleshov/llama-13b-3bit' # pulls from HF hub
llm = AutoLLMForCausalLM.from_pretrained(model_name).to('cuda')
tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-13b')

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

LLMTools comes with a patched version of the PEFT library that can be used to finetune the quantized models using the LP-LoRA method.

```python
from transformers import AutoTokenizer
from llmtools.llms.autollm import AutoLLMForCausalLM
from llmtools.engine.lora.peft import quant_peft

# load model and tokenizer
model_name = 'kuleshov/llama-13b-3bit' # pulls from HF hub
llm = AutoLLMForCausalLM.from_pretrained(model_name).to('cuda')
tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-13b')

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
trainer = transformers.Trainer(
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

LLMTools requires a UNIX environment supporting Python (3.8 or greater) and PyTorch (we tested with 1.13.1+cu116). See `requirements.txt` for details.

To ensure maximum reproducibility, consider creating a new conda environment:
```python
conda create -n llmtools
conda activate llmtools
conda install git pip virtualenv
```
LLMTools also requries an NVIDIA GPU (Pascal architecture or newer); other platforms are currently unsupported.

### Setup

We use `distutils` to package llmtools. If you are not running conda, you can also create a `virtualenv`.
```
pip install -r requirements.txt   # installs torch and two other packages
python setup.py install           # installs llmtools in your environment
```

Note that this process compiles and installs a custom CUDA kernel that is necessary to run quantized models.

## Running LLMTools

### Download and Quantize Models

First, start by downloading the weights of a base LLM model. Currently the LLAMA, OPT, and BLOOM are supported.
```python
from llmtools.llms.autollm import AutoLLMForCausalLM

model_name = 'decapoda-research/llama-7b-hf'
llm = AutoLLMForCausalLM.from_pretrained(model_name)
llm.eval()
```
You can quantize these models within `llmtools`.
```python
from llmtools.engine.quant.config import QuantConfig
from llmtools.engine.quant.gptq.executor import GPTQAlgorithm
from llmtools.data.calibration import get_calibration_loaders

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
```
You can save the quantized model to disk or to the HF hub.
```python
llm.save_pretrained(config.save)
print(f'Model weights saved to: {config.save}')
```

### Generation

Next, we generate text fron a quantized model. We first load the model.

```python
import torch
from transformers import AutoTokenizer
from llmtools.llms.autollm import AutoLLMForCausalLM

# load model and tokenizer
model_name = 'kuleshov/llama-13b-3bit' # pulls from HF hub
llm = AutoLLMForCausalLM.from_pretrained(model_name).to('cuda')
tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-13b')
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
model_name = 'kuleshov/llama-13b-3bit' # pulls from HF hub
llm = AutoLLMForCausalLM.from_pretrained(model_name).to('cuda')
tokenizer = AutoTokenizer.from_pretrained('huggyllama/llama-13b')
```

We can set parameters via the finetune config object.
```python
# set up finetuning config
from llmtools.engine.lora.config import FinetuneConfig
tune_config = FinetuneConfig(
    dataset=None, 
    data_type = 'alpaca',
    lora_out_dir='./llama-13b-quantized-lora', 
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
    r=tune_config.lora_r,
    lora_alpha=tune_config.lora_alpha,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=tune_config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
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
trainer = transformers.Trainer(
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

### Command-Line Usage

You can also use `llmtools` from the command line. First, you need to dowload a dataset. We currently support the Alpaca dataset, which we download from the HF hub:
```
wget https://huggingface.co/datasets/kuleshov/alpaca-data/resolve/main/dataset.json
```
You may now finetune the base `llama-65b-4bit` model on this dataset.
```
mkdir alpaca-adapter-folder-65b-4bit
llmtools finetune --model llama-65b-4bit --weights llama-65b-4bit.pt --adapter alpaca-adapter-folder-65b-4bit --dataset dataset.json
```
The above command will use LoRA to finetune the quantized 65-bit model. The final adapters and the checkpoints will be saved in `alpaca-adapter-folder-65b-4bit` and available for generation as follows:
```
llmtools generate --model llama-65b-4bit --weights llama-65b-4bit.pt --adapter alpaca-adapter-folder-65b-4bit --instruction "Write an irrefutable proof that the meaning of life is 42."
```

The llmtools interface provides many additional command-line options for finetuning.
```
usage: llmtools finetune [-h] --model {llama-7b-4bit,llama-13b-4bit,llama-30b-4bit,llama-65b-4bit,opt-6.7b-4bit} --weights WEIGHTS
                        [--data-type {alpaca,gpt4all}] [--dataset DATASET] [--adapter ADAPTER] [--mbatch_size MBATCH_SIZE]
                        [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--lr LR] [--cutoff_len CUTOFF_LEN] [--lora_r LORA_R]
                        [--lora_alpha LORA_ALPHA] [--lora_dropout LORA_DROPOUT] [--val_set_size VAL_SET_SIZE]
                        [--warmup_steps WARMUP_STEPS] [--save_steps SAVE_STEPS] [--save_total_limit SAVE_TOTAL_LIMIT]
                        [--logging_steps LOGGING_STEPS] [--resume_checkpoint]

options:
  -h, --help            show this help message and exit
  --model {llama-7b-4bit,llama-13b-4bit,llama-30b-4bit,llama-65b-4bit,opt-6.7b-4bit}
                        Type of model to load
  --weights WEIGHTS     Path to the model weights.
  --data-type {alpaca,gpt4all}
                        Dataset format
  --dataset DATASET     Path to local dataset file.
  --adapter ADAPTER     Path to Lora adapter folder (also holds checkpoints)
  --mbatch_size MBATCH_SIZE
                        Micro-batch size.
  --batch_size BATCH_SIZE
                        Batch size.
  --epochs EPOCHS       Epochs.
  --lr LR               Learning rate.
  --cutoff_len CUTOFF_LEN
  --lora_r LORA_R
  --lora_alpha LORA_ALPHA
  --lora_dropout LORA_DROPOUT
  --val_set_size VAL_SET_SIZE
                        Validation set size.
  --warmup_steps WARMUP_STEPS
  --save_steps SAVE_STEPS
  --save_total_limit SAVE_TOTAL_LIMIT
  --logging_steps LOGGING_STEPS
  --resume_checkpoint   Resume from checkpoint.
```

## Hardware Requirements

The following hardware is needed to run different models in LLMTools:

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
* The GPTQ algorithm and codebase by the [IST-DASLAB](https://github.com/IST-DASLab/gptq) with modifications by [@qwopqwop200](https://github.com/qwopqwop200/)
* The `alpaca_lora_4bit` repo by [johnsmith0031](https://github.com/johnsmith0031)
* The PEFT repo and its implementation of LoRA
* The LLAMA, OPT, and BLOOM models by META FAIR and the BigScience consortium

## Citations

Please cite this repository if you use our code.

```
@misc{LLMTools,
  author = {Volodymyr Kuleshov},
  title = {LLMTools: Fine-Tuning Large Language Models on One Consumer GPU},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/kuleshov-group/LLMTools}},
}
```

We also recommend you cite the above projects on which this work is based.

## Feedback

Please send feedback to [Volodymyr Kuleshov](https://twitter.com/volokuleshov).
