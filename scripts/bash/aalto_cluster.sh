#!/bin/bash -l
#SBATCH --job-name=gpu_run
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu-debug
#SBATCH --cpus-per-task=1
#SBATCH --chdir=/home/koutchc1/EduDPO
#SBATCH --output=/home/koutchc1/EduDPO/logs/%A_%a_gpu.log

export TRANSFORMERS_OFFLINE=1
export PATH=/home/koutchc1/.local/bin:$PATH
export PYTHONPATH="$HOME/EduDPO"

module load model-huggingface

module load mamba;
source activate eaai;

python3 scripts/run_sft.py 