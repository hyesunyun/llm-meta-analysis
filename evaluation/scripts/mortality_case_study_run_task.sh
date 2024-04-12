#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=case
#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --partition=short
#SBATCH --output=exec.case.%j.evaluate.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yun.hy@northeastern.edu

module load cuda/12.1
module load anaconda3/2022.05

source activate llm-meta-analysis

export $(xargs < ../../.env)

# gpt 4 turbo
python3 ../run_task.py --model gpt4 --task end_to_end --input_path /scratch/yun.hy/llm-meta-analysis/evaluation/data/meta_analysis_case_study.json --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/end_to_end --pmc_files_path /scratch/yun.hy/llm-meta-analysis/evaluation/data/no_attributes_case_study_markdown_files

# gpt 3.5 turbo
python3 ../run_task.py --model gpt35 --task end_to_end --input_path /scratch/yun.hy/llm-meta-analysis/evaluation/data/meta_analysis_case_study.json --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/end_to_end --pmc_files_path /scratch/yun.hy/llm-meta-analysis/evaluation/data/no_attributes_case_study_markdown_files

# mistral 7B
# python3 ../run_task.py --model mistral7B --task end_to_end --input_path /scratch/yun.hy/llm-meta-analysis/evaluation/data/meta_analysis_case_study.json --output_path /scratch/yun.hy/llm-meta-analysis/evaluation/outputs/end_to_end --pmc_files_path /scratch/yun.hy/llm-meta-analysis/evaluation/data/no_attributes_case_study_markdown_files
