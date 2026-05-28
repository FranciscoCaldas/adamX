#!/bin/bash
#
#SBATCH -p phd
#SBATCH --job-name=cexp_nc
#SBATCH --output=output/out_baseline_%j.txt
#SBATCH --gpus=1g.10g:1
#SBATCH -c 4

# SET MINICONDA_PATH HERE
MINICONDA_PATH=/home/f.caldas/miniconda3/ #EX:/home/<USER>/miniconda3/ (without any leading or trailing spaces)


export KMP_DUPLICATE_LIB_OK=TRUE
if [ -z "$MINICONDA_PATH" ]
then
	self=$(basename "$0")
	echo "JOB SUBMISSION FAILED. PLEASE SET MINICONDA_PATH on $self"
else
	source "$MINICONDA_PATH"etc/profile.d/conda.sh
	conda activate /data/f.caldas/miniconda3/envs/adamx
	echo "TASK: $TASK MODEL: $MODEL DATASET: $DATASET OPTIMIZER: $optimizer NUM_RUNS: $NUM_RUNS"
	srun python scripts/baseline_search.py --dataset cifar10 --optimizers yogi sgd rmsprop lion amsgrad
fi
