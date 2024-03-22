#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=biomistral
#SBATCH --cpus-per-task=8
#SBATCH --mem=75G
#SBATCH --partition=frink
#SBATCH --gres=gpu:1
#SBATCH --output=exec.biomistral.%j.evaluate.out

module load cuda/12.1
module load anaconda3/2022.05

source activate llm-meta-analysis

export $(xargs < ../../.env)

# json
python3 ../run_task.py --model biomistral --task binary_outcomes --split dev --prompt json --output_path evaluation/outputs/binary_outcomes --test

# yaml
python3 ../run_task.py --model biomistral --task binary_outcomes --split dev --prompt yaml --output_path evaluation/outputs/binary_outcomes --test