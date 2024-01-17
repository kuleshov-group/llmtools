#!/bin/bash
#SBATCH -J samsum7b-2bit-train-eval    # Job name
#SBATCH -o slurm_out/samsum-llama1-13b-2bit-train-eval.o%j    # Name of stdout output file (%j expands to jobId)
#SBATCH -e slurm_out/samsum-llama1-13b-2bit-train-eval.e%j    # Name of stderr output file
#SBATCH -N 1   # Total number of CPU nodes requested
#SBATCH -n 16   # Total number of CPU cores requrested
#SBATCH --mem=80gb    # CPU Memory pool for all cores
#SBATCH -t 72:00:00    # Run time (hh:mm:ss)
#SBATCH --partition=kuleshov --gres=gpu:a6000:1   # Which queue to run on, and what resources to use
                                               # --partition=<queue> - Use the `<queue>` queue
                                               # --gres=gpu:1 - Use 1 GPU of any type
                                               # --gres=gpu:1080ti:1 - Use 1 GTX 1080TI GPU

nvidia-smi

#export CUDA_HOME=/usr/local/cuda-11.2

cd /share/kuleshov/jy928/llmtools-2bit-e8/quip/quiptools && conda run -p /share/kuleshov/jy928/envs/llmtools-quip-0 --no-capture-output python setup.py install
cd /share/kuleshov/jy928/llmtools-2bit-e8 && conda run -p /share/kuleshov/jy928/envs/llmtools-quip-0 --no-capture-output python setup.py install
cd /share/kuleshov/jy928/llmtools-2bit-e8/experiment_samsum && conda run -p /share/kuleshov/jy928/envs/llmtools-quip-0 --no-capture-output python samsum_quip/train_samsum_llama_quip.py --model_name relaxml/Llama-1-13b-E8P-2Bit --adapter llama1_adapters/llama1-samsum-e8-7b-fix-seed42-mb4 --seed 42 --mbatch_size 4
cd /share/kuleshov/jy928/llmtools-2bit-e8/experiment_samsum && conda run -p /share/kuleshov/jy928/envs/llmtools-quip-0 --no-capture-output python samsum_quip/eval_samsum_llama_quip_debug.py --model_name relaxml/Llama-1-13b-E8P-2Bit --adapter /share/kuleshov/jy928/llmtools-2bit-e8/experiment_samsum/llama1_adapters/llama1-samsum-e8-7b-fix-seed42-mb4 --seed 42 --file_name llama1_samsum_e8_7b1_fix_42.txt --start_index 0 --end_index 819 --checkpoint_name /share/kuleshov/jy928/llmtools-2bit-e8/experiment_samsum/samsum_output/llama1-quip-e8-7b-fix-output-seed42


