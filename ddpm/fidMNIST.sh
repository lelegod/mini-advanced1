#!/bin/bash
#BSUB -J fid
# Run on A100 cluster BSUB -q gpua100
#BSUB -q gpuv100
#BSUB -W 1:00
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
#BSUB -o 02460_advanced_ml/week3/out/fid_%J.out
#BSUB -e 02460_advanced_ml/week3/out/fid_%J.err

nvidia-smi
# Load the cuda module
module load cuda/11.6
/appl/cuda/11.6.0/samples/bin/x86_64/linux/release/deviceQuery

source /zhome/f2/5/213953/02460_advanced_ml/.venv/bin/activate
python 02460_advanced_ml/week3/fid.py --model 02460_advanced_ml/week3/out/mnist.pt --device cuda --num_samples 128 --cls 02460_advanced_ml/week3/mnist_classifier.pth