# LLMTune: 4-Bit Finetuning of LLMs on a Consumer GPU

LLMTune allows finetuning LLMs (e.g., the largest 65B LLAMA models) on as little as one consumer-grade GPU.

Its features include:

* Modular support for multiple LLMs (currently LLAMA, OPT)
* Support for a wide range of consumer-grade NVidia GPUs; 65B LLAMAs finetune on one A6000
* Tiny and easy-to-use codebase

One benefit of being able to finetune larger LLMs (e.g., 65B params) on one GPU is the ability to easily leverage data parallelism for large models.

Underneath the hood, LLMTune implements the LoRA algorithm over an LLM compressed using the GPTQ algorithm, which requires implementing a backward pass for the quantized LLM. See the hardware requirements for more information on which LLMs are supported by various GPUs.

### Goals

LLMTune is a research project at Cornell Tech and Cornell University. Its goals are to:
* Provide an easy-to-use platform for creative experimentation with large language models
* Faciliate research on LLM alignment, bias mitigation, efficient inference, and other topics

## Demo

This is LLMTune running an instruction finetuned LLAMA-65B model on an NVidia A6000:

```
$ llmtune generate --model llama-65b-4bit --weights llama65b-4bit.pt --adapter alpaca-lora-65b-4bit --prompt "Write a well-thought out abstract for a machine learning paper that proves that 42 is the optimal seed for training neural networks."

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

This example is based on an Alpaca demo prompt. See below for additional examples.

## Installation

### Requirements

LLMTune requires a UNIX environment supporting Python (3.8 or greater) and PyTorch (we tested with 1.13.1+cu116). See `requirements.txt` for details.

To ensure maximum reproducibility, consider creating a new conda environment:
```
conda create -n llmtune
conda activate llmtune
conda install git pip virtualenv
```
LLMTune also requries an NVIDIA GPU (Pascal architecture or newer); other platforms are currently unsupported.

### Setup

We use `distutils` to package LLMTune. If you are not running conda, you can also create a `virtualenv`.
```
pip install -r requirements.txt   # installs torch and two other packages
python setup.py install           # installs llmtune in your environment
```

Note that this process compiles and installs a custom CUDA kernel that is necessary to run quantized models.

## Running LLMTune

The above process installs a `llmtune` command in your environment.

### Download Models

First, start by downloading the weights of a base LLM model:
```
wget https://huggingface.co/kuleshov/llama-65b-4bit/resolve/main/llama-65b-4bit.pt
```
The pre-quantized models are available for download. We will add the quantization code to `llmtune` if there is demand.
```
wget https://huggingface.co/kuleshov/llama-13b-4bit/resolve/main/llama-13b-4bit.pt
wget https://huggingface.co/kuleshov/llama-30b-4bit/resolve/main/llama-30b-4bit.pt
wget https://huggingface.co/kuleshov/llama-65b-4bit/resolve/main/llama-65b-4bit.pt
```
You can finetune these models yourself, or you can optionally download LoRA adapter weights that have already been finetuned for you using `llmtune`.
```
mkdir alpaca-adapter-65b-4bit && cd alpaca-adapter-65b-4bit
wget https://huggingface.co/kuleshov/alpaca-adapter-65b-4bit/resolve/main/adapter_config.json
wget https://huggingface.co/kuleshov/alpaca-adapter-65b-4bit/resolve/main/adapter_model.bin
```

### Generate Text

You can generate text directly from the command line. This generates text from the base model:
```
llmtune generate --model llama-65b-4bit --weights llama-65b-4bit.pt --prompt "the pyramids were built by"
```
More interestingly, we can generate output from an instruction-finetuned model by also providing a path to LoRA adapter weights. 
```
llmtune generate --model llama-65b-4bit --weights llama-65b-4bit.pt --adapter alpaca-adapter-65b-4bit --instruction "Write a well-thought out recipe for a blueberry lasagna dish." --max-length 500
```
In the above example, `--instruct` applies the Alpaca-style prompt template, although you can also use `--prompt` to feed the model initial text without any pre-processing:

The LLMTune interface also provides additional command-line options.
```
usage: llmtune generate [-h] --model {llama-7b-4bit,llama-13b-4bit,llama-30b-4bit,llama-65b-4bit,opt-6.7b-4bit} --weights WEIGHTS
                        [--adapter ADAPTER] [--prompt PROMPT] [--instruction INSTRUCTION] [--min-length MIN_LENGTH]
                        [--max-length MAX_LENGTH] [--top_p TOP_P] [--top_k TOP_K] [--temperature TEMPERATURE]

options:
  -h, --help            show this help message and exit
  --model {llama-7b-4bit,llama-13b-4bit,llama-30b-4bit,llama-65b-4bit,opt-6.7b-4bit}
                        Type of model to load
  --weights WEIGHTS     Path to the base model weights.
  --adapter ADAPTER     Path to the folder with the Lora adapter.
  --prompt PROMPT       Text used to initialize generation
  --instruction INSTRUCTION
                        Instruction for an alpaca-style model
  --min-length MIN_LENGTH
                        Minimum length of the sequence to be generated.
  --max-length MAX_LENGTH
                        Maximum length of the sequence to be generated.
  --top_p TOP_P         Top p sampling parameter.
  --top_k TOP_K         Top p sampling parameter.
  --temperature TEMPERATURE
                        Sampling temperature.
```

### Finetune A Base Model

You may also finetune a base model yourself. First, you need to dowload a dataset. We currently support the Alpaca dataset, which we download from the HF hub:
```
wget https://huggingface.co/datasets/kuleshov/alpaca-data/resolve/main/dataset.json
```
You may now finetune the base `llama-65b-4bit` model on this dataset.
```
mkdir alpaca-adapter-folder-65b-4bit
llmtune finetune --model llama-65b-4bit --weights llama-65b-4bit.pt --adapter alpaca-adapter-folder-65b-4bit --dataset dataset.json
```
The above command will use LoRA to finetune the quantized 65-bit model. The final adapters and the checkpoints will be saved in `alpaca-adapter-folder-65b-4bit` and available for generation as follows:
```
llmtune generate --model llama-65b-4bit --weights llama-65b-4bit.pt --adapter alpaca-adapter-folder-65b-4bit --instruction "Write an irrefutable proof that the meaning of life is 42."
```

The LLMTune interface provides many additional command-line options for finetuning.
```
usage: llmtune finetune [-h] --model {llama-7b-4bit,llama-13b-4bit,llama-30b-4bit,llama-65b-4bit,opt-6.7b-4bit} --weights WEIGHTS
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

### Programmatic Usage

LLMTune can also be used as a Python library:
```
import llmtune.executor as llmtune

llm, llm_config = llmtune.load_llm('llama-7b-4bit', '/path/to/llama-7b-4bit.pt')
output = llmtune.generate(
    llm, 
    llm_config, 
    prompt="the pyramids were built by", 
    min_length=10, 
    max_length=50, 
    top_p=0.95, 
    temperature=0.8,
)
print(output)
```

## Hardware Requirements

The following hardware is needed to run different models in LLMTune:

| Model Size | GPU Memory Requirements | Compatible GPUs |
| ----- | -------------------- | --------------- |
| 7b-4bit | 6GB | RTX 2060, 3050, 3060 |
| 13b-4bit | 10GB | GTX 1080, RTX 2060, 3060, 3080 |
| 30b-4bit | 20GB |  RTX 3080, A5000, 3090, 4090, V100 |
| 65b-4bit | 40GB | A100, 2x3090, 2x4090, A40, A6000 |

Only NVIDIA GPUs with the Pascal architecture or newer can run the current system.

## Additional Examples

In this example, the LLM produces a recipe for blueberry lasagna:
```
$ llmtune generate --model llama-65b-4bit --weights llama-65b-4bit.pt --adapter alpaca-lora-65b-4bit-e3 --instruction "Write a well-thought out recipe for a new blueberry lasagna dish." --max-length 500
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
$ llmtune generate --model llama-65b-4bit --weights /share/kuleshov/vk379/llama-65b-4bit.pt --prompt "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now? A: Let's think step-by-step."
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
* Make it easy to load models directly from the HF hub
* Out-of-the-box support for additional LLMs
* Improve the interface with things like automatic termination
* Automated quantization scripts

## Acknowledgements

LLMTune is based on the following projects:
* The GPTQ algorithm and codebase by the [IST-DASLAB](https://github.com/IST-DASLab/gptq) with modifications by [@qwopqwop200](https://github.com/qwopqwop200/)
* The `alpaca_lora_4bit` repo by [johnsmith0031](https://github.com/johnsmith0031)
* The PEFT repo and its implementation of LoRA
* The LLAMA, OPT, and BLOOM models by META FAIR and the BigScience consortium

## Citations

Please cite this repository if you use our code.

```
@misc{llmtune,
  author = {Volodymyr Kuleshov},
  title = {LLMTune: Fine-Tuning Large Language Models on One Consumer GPU},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/kuleshov-group/llmtune}},
}
```

We also recommend you cite the above projects on which this work is based.

## Feedback

Please send feedback to [Volodymyr Kuleshov](https://twitter.com/volokuleshov).
