#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --gres=gpu:v100:1
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G

source /home/mila/h/haolun.wu/projects/EduDPO/myvenv/bin/activate
module load python/3.10
nvidia-smi

python run.py run_sft --training_config config/task/train/train_sft_v1.yaml

### Run inference
python run.py run_inference --train_folder model_output/sft_v1

### Run judging
python run.py run_judging --input_file data/inference/sft_v1_questions_ta_feedbacks_train.json

echo "Finished running inference and judging on all models" 