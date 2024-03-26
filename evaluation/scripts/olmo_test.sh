#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=olmo
#SBATCH --cpus-per-task=8
#SBATCH --mem=75G
#SBATCH --partition=frink
#SBATCH --gres=gpu:1
#SBATCH --output=exec.olmo.%j.evaluate.out

module load cuda/12.1
module load anaconda3/2022.05

source activate llm-meta-analysis

export $(xargs < ../../.env)

python3 ../run_task.py --model olmo7B --task binary_outcomes --split dev --output_path evaluation/outputs/binary_outcomes --test