#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=biomistral
#SBATCH --cpus-per-task=8
#SBATCH --mem=75G
#SBATCH --partition=frink
#SBATCH --gres=gpu:1
#SBATCH --output=exec.biomistral.%j.evaluate.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yun.hy@northeastern.edu

# module load cuda/12.1
# module load anaconda3/2022.05

conda activate llm-meta-analysis

export $(xargs < ../../.env)

# outcome_types
python3 ../run_task.py --model biomistral --task outcome_type --split test --prompt without-abstract-results --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/outcome_type

# binary outcomes
# python3 ../run_task.py --model biomistral --task binary_outcomes --split test --prompt yaml --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/binary_outcomes

# continuous outcomes
# python3 ../run_task.py --model biomistral --task continuous_outcomes --split test --prompt yaml --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/continuous_outcomes