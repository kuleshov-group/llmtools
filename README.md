# LLMTune: 4-Bit Finetuning of Large Language Models on One Consumer GPU

LLMTune allows finetuning LLMs (including the largest 65B LLAMA models) on as little as one consumer-grade GPU.

Its features include:

* Modular support for multiple LLMs (currently LLAMA, OPT)
* Support for a wide range of consumer-grade Nvidia GPUs; 65B LLAMAs finetune on one A6000
* Tiny and easy-to-use codebase

One benefit of being able to finetune larger LLMs (e.g., 65B params) on one GPU is the ability to easily scale up training using only data parallelism.

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

This example is based on an Alpaca prompt. See below for additional examples.

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
virtualenv llmtune_env
source /llmtune_env/bin/activate
pip install -r requirements.txt   # installs torch and two other packages
python setup.py install           # installs llmtune in your environment
export CUDA_VISIBLE_DEVICES=0     # your GPU should be visible
```

Note that this process compiles and installs a custom CUDA kernel that is necessary to run quantized models.

## Running LLMTune

The above process installs a `llmtune` command in your environment.

### Download Models

First, start by downloading the weights of an LLM model:
```
llmtune download --model llama-7b-4bit --weights llama-7b-4bit.pt
```
You can also download the weights directly using `wget`:
```
wget https://huggingface.co/kuleshov/llama-30b-4bit/resolve/main/llama-30b-4bit.pt
wget https://huggingface.co/kuleshov/llama-65b-4bit/resolve/main/llama-65b-4bit.pt
```
The following models have pre-quantized weights: `llama-65b-4bit`,`llama-65b-4bit`, `llama-llmtuned-65b-4bit`.

### Generate Text

You can generate text directly from the command line:
```
llmtune generate --model llama-7b-4bit --weights llama-7b-4bit.pt --prompt "compose a haiku about rain"
```

The LLMTune interface also provides additional command-line options.
```
usage: llmtune generate [-h] --model {llama-7b-4bit,llama-13b-4bit} --weights WEIGHTS [--prompt PROMPT]
                        [--min-length MIN_LENGTH] [--max-length MAX_LENGTH] [--top_p TOP_P]
                        [--temperature TEMPERATURE]

options:
  -h, --help            show this help message and exit
  --model {llama-7b-4bit,llama-13b-4bit}
                        Type of model to load
  --weights WEIGHTS     Path to the model weights.
  --prompt PROMPT       Text used to initialize generation
  --min-length MIN_LENGTH
                        Minimum length of the sequence to be generated.
  --max-length MAX_LENGTH
                        Maximum length of the sequence to be generated.
  --top_p TOP_P         Top p sampling parameter.
  --temperature TEMPERATURE
                        Sampling temperature.
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

This is experimental work in progress. Things which still need work include:
* Make it easy to load models directly from the HF hub
* Out-of-the-box support for additional LLMs
* Automated quantization scripts

## Acknowledgements

LLMTune is based on the following projects:
* The GPTQ algorithm and codebase by the [IST-DASLAB](https://github.com/IST-DASLab/gptq) with modifications by [@qwopqwop200](https://github.com/qwopqwop200/)
* The `alpaca_lora_4bit` repo by [johnsmith0031](https://github.com/johnsmith0031)
* The LLAMA, OPT, and BLOOM models by META FAIR and the BigScience consortium.

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

Please send feedback to [Volodymyr Kuleshov](https://www.cs.cornell.edu/~kuleshov/).
