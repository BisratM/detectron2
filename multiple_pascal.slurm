#!/bin/bash
#SBATCH --job-name=multiplepascal       # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks=2               # total number of tasks across all nodes
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=06:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all          # send email on job start, end and fail
#SBATCH --mail-user=bmoges@princeton.edu

module purge
module load anaconda3
module load rh/devtoolset 
module load cudnn/cuda-10.1/7.5.0


conda activate pytorch 
pip install -e . 
srun python tools/train_net.py --num-gpus 8 --num-machines 2 --config-file configs/PascalVOC-Detection/simple_test.yaml
