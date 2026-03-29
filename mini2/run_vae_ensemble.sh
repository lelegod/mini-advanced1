#!/bin/sh
#BSUB -q gpuv100
#BSUB -J vae_geodesics
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -B
#BSUB -N
#BSUB -o output_%J.out
#BSUB -e error_%J.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate advanced_machine_learning

python mini2/geodesics/ensemble_vae.py geodesics \
    --experiment-folder "experiment" \
    --device cuda \
    --num-curves 25

python mini2/geodesics/ensemble_vae.py evaluate_cov \
    --experiment-folder "experiment" \
    --device cuda \
    --num-reruns 10 \
    --num-decoders 3 \
    --num-curves 10 \
    --epochs-per-decoder 50
