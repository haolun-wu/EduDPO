#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --partition=short-unkillable
#SBATCH --gres=gpu:a100l:4
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source /home/mila/h/haolun.wu/projects/plugin-decoding/statml/bin/activate
module load python/3.10
nvidia-smi

python run.py run_dpo

echo "Finished running DPO"