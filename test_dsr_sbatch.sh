#!/bin/sh
#SBATCH --job-name=test_dsr
#SBATCH --output=./outputs/test_dsr.out
#SBATCH --time=01:00:00       # job time limit
#SBATCH --nodes=1             # number of nodes
#SBATCH --ntasks-per-node=4            # number of tasks
#SBATCH --partition=gpu       # partition to run on nodes that contain gpus
#SBATCH --cpus-per-task=16     # number of allocated cores
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=64G     # memory allocation

module purge
module load Anaconda3/2023.07-2
source activate your_conda_env

MVTEC3D_PATH=/path/to/mvtec3d/
CHKP_PATH=/path/to/checkpoints/
RUN_NAME=3dsr_MODEL


echo $RUN_NAME

srun --nodes=1 --exclusive --gres=gpu:1 --ntasks=1 --unbuffered python test_dsr.py --gpu_id 0 --data_path $MVTEC3D_PATH --out_path $OUT_PATH --run_name $RUN_NAME &

wait
