#!/bin/bash
#SBATCH --job-name=deepmask_training       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=16G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=00:02:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=dl13@princeton.edu

module purge
module load anaconda3
conda activate tf-gpu
conda activate deepmask
cd deepmask-pytorch
DEEPMASK=$PWD
export PYTHONPATH=$DEEPMASK:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/train.py --dataset coco -j 20 --freeze_bn
