#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=gemma
#SBATCH --mem=20G
#SBATCH --partition=short
#SBATCH --output=exec.gemma-eval.%j.evaluate.out

module load anaconda3/2022.05

source activate llm-meta-analysis

export $(xargs < ../../.env)

# outcome_types
# python3 ../evaluate_output.py --task outcome_type --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/outcome_type/gemma7B_outcome_type_test_output_20240326-12:00:55.json --metrics_path /scratch/yun.hy/llm-meta-analysis/evaluation/metrics/outcome_type/

# binary outcomes
python3 ../evaluate_output.py --task binary_outcomes --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/binary_outcomes/gemma7B_binary_outcomes_test_output_20240403-13:36:04.json --metrics_path /scratch/yun.hy/llm-meta-analysis/evaluation/metrics/binary_outcomes/

# continuous_outcomes
python3 ../evaluate_output.py --task continuous_outcomes --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/continuous_outcomes/gemma7B_continuous_outcomes_test_output_20240403-14:40:46.json --metrics_path /scratch/yun.hy/llm-meta-analysis/evaluation/metrics/continuous_outcomes/
