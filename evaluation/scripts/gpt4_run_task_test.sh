#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=64:00:00
#SBATCH --job-name=gpt4
#SBATCH --cpus-per-task=4
#SBATCH --mem=45G
#SBATCH --partition=frink
#SBATCH --output=exec.gpt4.%j.evaluate.out

module load anaconda3/2022.05

source activate llm-meta-analysis

export $(xargs < ../../.env)

# outcome_types
python3 ../run_task.py --model gpt4 --task outcome_types --split dev --prompt without-abstract-results --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/outcome_types --test