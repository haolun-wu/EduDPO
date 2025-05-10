#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --gres=gpu:v100:1
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G

source /home/mila/h/haolun.wu/projects/EduDPO/myvenv/bin/activate
module load python/3.10
nvidia-smi

# Run inference on SFT model
python run.py run_inference --train_folder sft_output/sft_v1
# python run.py run_judging --input_file data/inference/sft_v1_questions_ta_feedbacks_train.json

# # Run inference on DPO models
python run.py run_inference --train_folder dpo_output/dpo
# python run.py run_judging --input_file data/inference/dpo_questions_ta_feedbacks_train.json

python run.py run_inference --train_folder dpo_output/rpo
# python run.py run_judging --input_file data/inference/rpo_questions_ta_feedbacks_train.json

python run.py run_inference --train_folder dpo_output/slic_beta_0.01
# python run.py run_judging --input_file data/inference/slic_beta_0.01_questions_ta_feedbacks_train.json

python run.py run_inference --train_folder dpo_output/slic_beta_0.05
# python run.py run_judging --input_file data/inference/slic_beta_0.05_questions_ta_feedbacks_train.json

echo "Finished running inference and judging on all models" 