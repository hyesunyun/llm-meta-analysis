#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=case
#SBATCH --mem=20G
#SBATCH --partition=short
#SBATCH --output=exec.case-eval.%j.evaluate.out

module load anaconda3/2022.05

source activate llm-meta-analysis

export $(xargs < ../../.env)

# outcome_types
python3 ../evaluate_output.py --task outcome_type --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/end_to_end/mistral7B_outcome_type_None_output_20240404-11:54:12.json --metrics_path /scratch/yun.hy/llm-meta-analysis/evaluation/metrics/end_to_end/
python3 ../evaluate_output.py --task outcome_type --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/end_to_end/gpt35_outcome_type_None_output_20240404-13:18:05.json --metrics_path /scratch/yun.hy/llm-meta-analysis/evaluation/metrics/end_to_end/

# binary outcomes
python3 ../evaluate_output.py --task binary_outcomes --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/end_to_end/mistral7B_binary_outcomes_None_output_20240404-12:01:28.json --metrics_path /scratch/yun.hy/llm-meta-analysis/evaluation/metrics/end_to_end/
python3 ../evaluate_output.py --task binary_outcomes --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/end_to_end/gpt35_binary_outcomes_None_output_20240404-13:18:18.json --metrics_path /scratch/yun.hy/llm-meta-analysis/evaluation/metrics/end_to_end/