#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=75G
#SBATCH --partition=frink
#SBATCH --gres=gpu:1
#SBATCH --output=exec.test.%j.evaluate.out

# module load cuda/12.1
# module load anaconda3/2022.05

# source activate llm-meta-analysis

# export $(xargs < ../../.env)

# python3 ../run_task.py --model gemma7B --task continuous_outcomes --split dev --output_path evaluation/outputs/continuous_outcomes --test

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

# module load discovery/2021-10-06
# module load cuda/12.1
# module load anaconda3/2022.05

source ~/.bashrc
conda activate llm-meta-analysis

export $(xargs < ../../.env)
# continuous outcomes
python3 ../run_task.py --model pmc-llama --task continuous_outcomes --split dev --prompt yaml --output_path test_outputs --test
