#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --gres=gpu:v100:1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G

source /home/mila/h/haolun.wu/projects/EduDPO/myvenv/bin/activate
module load python/3.10
nvidia-smi

### Run inference
python run.py run_inference --train_folder model_output/dpo

### Run judging
python run.py run_judging --input_file data/inference/dpo_questions_ta_feedbacks_train.json

echo "Finished running inference and judging on all models" 