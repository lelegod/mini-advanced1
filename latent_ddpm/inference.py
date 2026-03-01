import torch
import os
from torchvision.utils import save_image
from torchvision import datasets, transforms

from config import M, BETA_VAE, NUM_HIDDEN, T
from beta_VAE import get_vae_model 
from ddpm import DDPM, FcNetwork 
from fid import compute_fid
from config import M, BETA_VAE, NUM_HIDDEN, T, DDPM_BETA_1, DDPM_BETA_T

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    vae_model = get_vae_model(M, BETA_VAE, device)
    vae_model.load_state_dict(torch.load('beta_1/beta_vae.pt', map_location=device))
    vae_model.eval() 
    print("Successfully loaded beta_vae.pt")

    network = FcNetwork(input_dim=M, num_hidden=NUM_HIDDEN)
    latent_ddpm = DDPM(network, DDPM_BETA_1, DDPM_BETA_T, T=T).to(device)
    latent_ddpm.load_state_dict(torch.load('beta_1/latent_ddpm.pt', map_location=device))
    latent_ddpm.eval()
    print("Successfully loaded latent_ddpm.pt")

    stats = torch.load('beta_1/latent_stats.pt', map_location=device)
    z_mean, z_std = stats['mean'], stats['std']

    num_samples_grid = 1
    
    with torch.no_grad():
        sampled_z = latent_ddpm.sample((num_samples_grid, M)).to(device)
        sampled_z = (sampled_z * z_std) + z_mean
        generated_images = vae_model.decoder(sampled_z).mean
        
        generated_images = generated_images.view(num_samples_grid, 1, 28, 28)
        generated_images = (generated_images / 2) + 0.5
        save_image(generated_images, 'samples/sample_pic.png', nrow=2)
        print(f"Saved {num_samples_grid} samples to sample_grid_2x2.png")

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])
    
    # data_path = os.path.join('..', 'data')
    # test_dataset = datasets.MNIST(
    #     root=data_path, 
    #     train=False, 
    #     download=True,
    #     transform=transform
    # )
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # # Get all real test images
    # real_images_list = []
    # for images, _ in test_loader:
    #     real_images_list.append(images)
    # real_images = torch.cat(real_images_list, dim=0)
    # print(f"Loaded {real_images.shape[0]} real test images")

    # num_samples_fid = 5000
    # print(f"\nGenerating {num_samples_fid} samples for FID computation...")
    
    # generated_images_fid = []
    # batch_size_gen = 256
    
    # with torch.no_grad():
    #     for batch_start in range(0, num_samples_fid, batch_size_gen):
    #         batch_size = min(batch_size_gen, num_samples_fid - batch_start)
    #         sampled_z = latent_ddpm.sample((batch_size, M)).to(device)
    #         sampled_z = (sampled_z * z_std) + z_mean
    #         gen_images = vae_model.decoder(sampled_z).mean
    #         gen_images = gen_images.view(batch_size, 1, 28, 28)
            
    #         generated_images_fid.append(gen_images.cpu())
    #         if (batch_start + batch_size) % 1000 == 0:
    #             print(f"  Generated {batch_start + batch_size}/{num_samples_fid} samples")
    
    # generated_images_all = torch.cat(generated_images_fid, dim=0)
    # print(f"Generated {generated_images_all.shape[0]} samples for FID")

    # print("\nComputing FID score...")
    # fid_score = compute_fid(
    #     real_images.to(device),
    #     generated_images_all.to(device),
    #     device=device,
    #     classifier_ckpt='mnist_classifier.pth'
    # )
    
    # print(f"FID Score: {fid_score:.4f}")
    
    # # Save FID result to file
    # with open('fid_results.txt', 'w') as f:
    #     f.write(f"FID Score: {fid_score:.4f}\n")
    #     f.write(f"Real samples: {real_images.shape[0]}\n")
    #     f.write(f"Generated samples: {generated_images_all.shape[0]}\n")
    
    # print("FID results saved to fid_results.txt")