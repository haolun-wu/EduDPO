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

# python run.py run_data_full_pipeline
# python run.py run_dpo_data_process

# python run.py run_dpo --training_config config/task/train/train_dpo.yaml
# python run.py run_dpo --training_config config/task/train/train_rpo.yaml
# python run.py run_dpo --training_config config/task/train/train_slic_beta_0.01.yaml
# python run.py run_dpo --training_config config/task/train/train_slic_beta_0.05.yaml
python run.py run_sft --training_config config/task/train/train_sft_v1.yaml

echo "Finished running DPO"