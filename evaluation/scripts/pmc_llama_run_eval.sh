#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=pmc-llama
#SBATCH --mem=20G
#SBATCH --partition=short
#SBATCH --output=exec.pmc-llama-eval.%j.evaluate.out

module load anaconda3/2022.05

source activate llm-meta-analysis

export $(xargs < ../../.env)

# outcome_types
# python3 ../evaluate_output.py --task outcome_type --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/outcome_type/pmc-llama_outcome_type_test_output_20240326-14:21:40.json --metrics_path /scratch/yun.hy/llm-meta-analysis/evaluation/metrics/outcome_type/

# binary outcomes
python3 ../evaluate_output.py --task binary_outcomes --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/binary_outcomes/xml/pmc-llama_binary_outcomes_test_output_20240331-22:04:16.json --metrics_path /scratch/yun.hy/llm-meta-analysis/evaluation/metrics/binary_outcomes/xml/

# continuous_outcomes
python3 ../evaluate_output.py --task continuous_outcomes --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/continuous_outcomes/xml/pmc-llama_continuous_outcomes_test_output_20240401-18:10:38.jsonpmc-llama_continuous_outcomes_test_output_20240401-18:10:38.json --metrics_path /scratch/yun.hy/llm-meta-analysis/evaluation/metrics/continuous_outcomes/xml/
