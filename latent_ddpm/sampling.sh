#!/bin/bash
#BSUB -J measure_time
#BSUB -q gpuv100
#BSUB -W 0:20
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -B
#BSUB -N
#BSUB -o measure_out_%J.out
#BSUB -e measure_err_%J.err

nvidia-smi

# Load CUDA
module load cuda/11.6

source /zhome/4b/9/223637/.conda/envs/advancedml/.venv/bin/activate

python measure_time.py --folder beta_1