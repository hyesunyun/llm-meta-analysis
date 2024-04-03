#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=test
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=75G
#SBATCH --partition=177huntington
#SBATCH --gres=gpu:1
#SBATCH --output=exec.test.%j.evaluate.out

source ~/.bashrc
conda activate llm-meta-analysis

export $(xargs < ../../.env)

python3 ../run_task.py --model gemma7B --task continuous_outcomes --split dev --output_path evaluation/outputs/continuous_outcomes
python3 ../run_task.py --model pmc-llama --task binary_outcomes --split dev --output_path evaluation/outputs/binary_outcomes
