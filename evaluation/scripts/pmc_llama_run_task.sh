#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=pmc-llama
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=75G
#SBATCH --partition=177huntington
#SBATCH --gres=gpu:1
#SBATCH --output=exec.pmc-llama.%j.evaluate.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yun.hy@northeastern.edu

source ~/.bashrc
conda activate llm-meta-analysis

export $(xargs < ../../.env)

# outcome type
# python3 ../run_task.py --model pmc-llama --task outcome_type --split test --prompt without-abstract-results --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/outcome_type

# binary outcomes
python3 ../run_task.py --model pmc-llama --task binary_outcomes --split test --prompt yaml --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/binary_outcomes

# continuous outcomes
# python3 ../run_task.py --model pmc-llama --task continuous_outcomes --split test --prompt yaml --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/continuous_outcomes