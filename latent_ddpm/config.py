"""
Configuration file for Latent DDPM model hyperparameters.
Update this file to change hyperparameters across all scripts.
"""

# VAE Hyperparameters
M = 32
BETA_VAE = 1e-6
VAE_LR = 1e-3
VAE_EPOCHS = 50
VAE_DECODER_STD = 0.1
VAE_PATIENCE = 3

# DDPM Hyperparameters
T = 1000
NUM_HIDDEN = 256
DDPM_BETA_1 = 1e-4
DDPM_BETA_T = 2e-2
DDPM_EPOCHS = 100
DDPM_PATIENCE = 100
DDPM_LR = 1e-3

# Data Hyperparameters
BATCH_SIZE = 128
TRAIN_SIZE = 50000
VAL_SIZE = 10000

# Inference/Generation Hyperparameters
NUM_SAMPLES = 64
