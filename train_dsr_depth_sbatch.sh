#!/bin/sh
#SBATCH --job-name=train_dsr_depth
#SBATCH --output=./outputs/train_dsr_depth.out
#SBATCH --time=1-23:30:00       # job time limit
#SBATCH --nodes=3             # number of nodes
#SBATCH --ntasks-per-node=4            # number of tasks
#SBATCH --partition=gpu       # partition to run on nodes that contain gpus
#SBATCH --cpus-per-task=16     # number of allocated cores
#SBATCH --gres=gpu:4
#SBATCH --mem-per-gpu=64G     # memory allocation

module purge
module load Anaconda3/2023.07-2
source activate your_conda_env

MVTEC3D_PATH=/path/to/mvtec3d/
CHKP_PATH=/output/path/for/checkpoints/
RUN_NAME=3dsr_depth_MODEL

# i marks the object id, the names of objects are in a list in the train_dsr_depth.py main
for i in $(seq 0 9)
do
    srun --nodes=1 --exclusive --gres=gpu:1 --ntasks=1 --unbuffered python train_dsr_depth.py --gpu_id 0 --obj_id $i --lr 0.0002 --bs 16 --epochs 120 --data_path $MVTEC3D_PATH --out_path $CHKP_PATH --run_name $RUN_NAME &
done


wait
