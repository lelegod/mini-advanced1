import torch
import argparse
import os
import numpy as np
from torchvision import datasets, transforms
from model import create_vae_model
from fid import compute_fid

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior', type=str, default='flow', choices=['gaussian', 'mog', 'flow'], help='Prior to use for VAE')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable')
    parser.add_argument('--n-samples', type=int, default=5000, metavar='N', help='number of samples for FID')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load binarized MNIST
    threshold = 0.5
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: (threshold < x).float().squeeze())
    ])

    mnist_test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/', train=False, download=True, transform=transform),
        batch_size=args.n_samples, shuffle=True
    )
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    classifier_ckpt = os.path.join(script_dir, "mnist_classifier.pth")

    # Define VAE model
    model = create_vae_model(args.prior, args.latent_dim, device)
    model_path = os.path.join(script_dir, f"model/{args.prior}/{args.prior}_vae_model_run0.pt")

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device), weights_only=True))
    model.eval()

    with torch.no_grad():
        x_real = next(iter(mnist_test_loader))[0].to(device).view(args.n_samples, 1, 28, 28)
        x_gen = model.sample(args.n_samples).view(args.n_samples, 1, 28, 28)
        
        # Scale to [-1, 1] as expected by fid.py
        x_real_scaled = x_real * 2.0 - 1.0
        x_gen_scaled = x_gen * 2.0 - 1.0
        
        fid_score = compute_fid(x_real_scaled, x_gen_scaled, device=device, classifier_ckpt=classifier_ckpt)
        
    print(f"FID Score for {args.prior.upper()} (Run 0): {fid_score:.4f}")
