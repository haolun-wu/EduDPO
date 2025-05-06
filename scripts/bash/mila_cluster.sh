#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:a100l:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

source /home/mila/h/haolun.wu/projects/EduDPO/myvenv/bin/activate
module load python/3.10
nvidia-smi

python run.py run_data_full_pipeline
python run.py run_dpo_data_process
python run.py run_dpo

echo "Finished running DPO"