#!/bin/bash
#
#SBATCH --job-name=example_job
#
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=piech
#SBATCH --account=piech
#SBATCH --mem=8G
#SBATCH --nodelist=piech1
#
#SBATCH --chdir=/sailhome/ckou/diagnostic
#SBATCH --output=/sailhome/ckou/diagnostic/logs/%A.log

echo “hello world”

