#!/bin/sh
#BSUB -q gpuv100
#BSUB -J vae_geodesics
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 04:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -B
#BSUB -N
#BSUB -o output_%J.out
#BSUB -e error_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate advanced_machine_learning

python mini2/geodesics/ensemble_vae.py geodesics \
    --experiment-folder "mini2/geodesics/experiment_partA" \
    --device cuda \
    --num-curves 25

python mini2/geodesics/ensemble_vae.py geodesics_ensemble \
    --experiment-folder "mini2/geodesics/partB" \
    --device cuda \
    --num-decoders 10 \
    --num-curves 25

python mini2/geodesics/ensemble_vae.py evaluate_cov \
    --experiment-folder "mini2/geodesics/experiment" \
    --device cuda \
    --num-reruns 10 \
    --num-decoders 10 \
    --num-curves 15 \
    --epochs-per-decoder 50
