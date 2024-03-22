#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=pmc_llama
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --partition=177huntington
#SBATCH --gres=gpu:1
#SBATCH --output=exec.pmcllama.%j.evaluate.out

module load cuda/12.1
module load anaconda3/2022.05

source activate llm-meta-analysis

export $(xargs < ../../.env)

# json
python3 ../run_task.py --model pmc-llama --task continuous_outcomes --split dev --prompt json --output_path evaluation/outputs/continuous_outcomes --test

# yaml
python3 ../run_task.py --model pmc-llama --task continuous_outcomes --split dev --prompt yaml --output_path evaluation/outputs/continuous_outcomes --test