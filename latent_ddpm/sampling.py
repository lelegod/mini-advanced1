import torch
import time
from time import perf_counter
import numpy as np
import sys
import os
import argparse

from config import M, BETA_VAE, NUM_HIDDEN, T, DDPM_BETA_1, DDPM_BETA_T
from beta_VAE import get_vae_model 
from ddpm import DDPM, FcNetwork 

def flush_print(msg):
    print(msg)
    sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help='Path to model folder')
    args = parser.parse_args()
    folder = args.folder

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    flush_print(f"Device: {device}")
    
    if not os.path.exists(folder):
        flush_print(f"Error: Folder '{folder}' not found.")
        sys.exit(1)

    # Load VAE
    vae_model = get_vae_model(M, BETA_VAE, device)
    vae_model.load_state_dict(torch.load(os.path.join(folder, 'beta_vae.pt'), map_location=device))
    vae_model.eval() 

    # Load Latent DDPM
    network = FcNetwork(input_dim=M, num_hidden=NUM_HIDDEN)
    latent_ddpm = DDPM(network, DDPM_BETA_1, DDPM_BETA_T, T=T).to(device)
    latent_ddpm.load_state_dict(torch.load(os.path.join(folder, 'latent_ddpm.pt'), map_location=device))
    latent_ddpm.eval()

    # Load stats
    stats = torch.load(os.path.join(folder, 'latent_stats.pt'), map_location=device)
    z_mean, z_std = stats['mean'], stats['std']

    num_samples = 5000
    timings = []

    if device == 'cuda':
        with torch.no_grad():
            _ = latent_ddpm.sample((1, M)).to(device)
            torch.cuda.synchronize()

    flush_print(f"Profiling {num_samples} generations...")
    
    with torch.no_grad():

        if device == 'cuda':
            torch.cuda.synchronize()
        

        start = perf_counter()
        sampled_z = latent_ddpm.sample((num_samples, M)).to(device)
        sampled_z = (sampled_z * z_std) + z_mean
        _ = vae_model.decoder(sampled_z).mean

        if device == 'cuda':
            torch.cuda.synchronize()
        
        end = perf_counter()

        total_time = end - start
        avg_time = total_time / num_samples

        flush_print(f"Total time for batch ({num_samples} samples): {total_time:.4f} s")
        flush_print(f"Average time per sample: {avg_time:.4f} s")
        flush_print(f"Throughput: {num_samples / total_time:.2f} samples/sec")