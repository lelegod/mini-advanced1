import torch
import torch.nn as nn
import argparse
import os
from torchvision.utils import save_image
from model import create_vae_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=str, default='gaussian', choices=['gaussian', 'mog', 'flow'], help='Prior to use for VAE')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Define VAE model
    model = create_vae_model(args.prior, args.latent_dim, device)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, f"model/{args.prior}/{args.prior}_vae_model.pt")

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=True))
    model.eval()
    with torch.no_grad():
        samples = (model.sample(64)).cpu() 
        samples_dir = os.path.join(script_dir, f"model/{args.prior}/samples")
        os.makedirs(samples_dir, exist_ok=True)
        
        reshaped_samples = samples.view(64, 1, 28, 28)
        for i in range(64):
            sample_path = os.path.join(samples_dir, f"{args.prior}_sample_{i}.png")
            save_image(reshaped_samples[i], sample_path)