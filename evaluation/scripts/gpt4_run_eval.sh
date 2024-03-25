#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=gpt4
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --partition=frink
#SBATCH --output=exec.gpt4-eval.%j.evaluate.out

module load anaconda3/2022.05

source activate llm-meta-analysis

export $(xargs < ../../.env)

# outcome_types
# python3 ../evaluate_output.py --task outcome_type --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/outcome_type/gpt4_outcome_type_test_output_20240325-11:55:44.json --metrics_path /scratch/yun.hy/llm-meta-analysis/evaluation/metrics/outcome_type/

# binary outcomes
python3 ../evaluate_output.py --task binary_outcomes --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/binary_outcomes/gpt4_binary_outcomes_test_output_20240325-12:19:11.json --metrics_path /scratch/yun.hy/llm-meta-analysis/evaluation/metrics/binary_outcomes/

# continuous_outcomes
python3 ../evaluate_output.py --task continuous_outcomes --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/continuous_outcomes/gpt4_continuous_outcomes_test_output_20240325-13:46:14.json --metrics_path /scratch/yun.hy/llm-meta-analysis/evaluation/metrics/continuous_outcomes