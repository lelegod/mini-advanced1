#!/bin/bash
#BSUB -J fid
# Run on A100 cluster BSUB -q gpua100
#BSUB -q gpuv100
#BSUB -W 1:00
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -o /zhome/af/5/222919/Documents/mini-advanced1/vae_flow/out/fid_%J.out
#BSUB -e /zhome/af/5/222919/Documents/mini-advanced1/vae_flow/out/fid_%J.err

# Change to project root
cd /zhome/af/5/222919/Documents/mini-advanced1
mkdir -p vae_flow/out

nvidia-smi
# Load the cuda module
module load cuda/11.6
/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate advanced_machine_learning
python vae_flow/sample.py --prior flow --device cpu --n-samples 2000000