#!/bin/sh
#SBATCH --job-name=train_dada_depth
#SBATCH --output=./outputs/train_dada_depth.out
#SBATCH --time=1-23:59:00       # job time limit
#SBATCH --nodes=1             # number of nodes
#SBATCH --ntasks-per-node=1            # number of tasks
#SBATCH --partition=gpu       # partition to run on nodes that contain gpus
#SBATCH --cpus-per-task=16     # number of allocated cores
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=64G     # memory allocation

module purge
module load Anaconda3/2023.07-2
source activate your_conda_env

srun --nodes=1 --exclusive --gres=gpu:1 --ntasks=1 --unbuffered python train_dada_depth.py 0 &

wait
