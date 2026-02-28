import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image

from config import M, BETA_VAE, TRAIN_SIZE
from beta_VAE import get_vae_model 

def run_reconstruction_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading VAE model...")
    vae_model = get_vae_model(M, BETA_VAE, device)
    
    try:
        vae_model.load_state_dict(torch.load('beta_vae.pt', map_location=device))
        print("Successfully loaded beta_vae.pt")
    except FileNotFoundError:
        print("Error: beta_vae.pt not found. Please make sure you have trained the VAE.")
        return

    vae_model.eval()

    print("Loading validation data...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),
        transforms.Lambda(lambda x: (x-0.5)*2.0),
        transforms.Lambda(lambda x: x.flatten())
    ])
    
    full_train_data = datasets.MNIST('data/', train=True, download=True, transform=transform)
    train_size = TRAIN_SIZE
    val_size = len(full_train_data) - train_size
    _, val_data = torch.utils.data.random_split(
        full_train_data, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=True)

    print("Running reconstruction test...")
    with torch.no_grad():
        real_images, _ = next(iter(val_loader))
        real_images = real_images.to(device)
        
        z_encoded = vae_model.encoder(real_images).sample()
        reconstructed_images = vae_model.decoder(z_encoded).mean
        
        real_images = real_images.view(-1, 1, 28, 28)
        reconstructed_images = reconstructed_images.view(-1, 1, 28, 28)
        
        comparison = torch.cat([real_images, reconstructed_images])
        
        # Save the image grid
        save_image(comparison, 'vae_reconstructions.png', nrow=8, normalize=True, value_range=(-1, 1))
        
        print("Done! Saved reconstructions to vae_reconstructions.png.")
        print("-> The top 4 rows are the REAL images.")
        print("-> The bottom 4 rows are the RECONSTRUCTIONS.")

if __name__ == "__main__":
    run_reconstruction_test()