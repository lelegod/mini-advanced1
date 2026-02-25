import torch
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from beta_VAE import get_vae_model 
from ddpm import DDPM, FcNetwork 

def train_latent_ddpm(vae, ddpm, optimizer, data_loader, epochs, device):
    ddpm.train()
    total_steps = len(data_loader) * epochs
    progress_bar = tqdm(range(total_steps), desc="Training Latent DDPM")

    for epoch in range(epochs):
        for x, _ in data_loader:
            x = x.to(device)
            
            # Encode image to latent space
            with torch.no_grad():
                z = vae.encoder(x).sample() 
            
            # Train DDPM on the latent vector z
            optimizer.zero_grad()
            loss = ddpm.loss(z) 
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    M = 32
    beta = 1e-6
    ddpm_epochs = 50 
    batch_size = 64

    # Load the Frozen VAE 
    vae_model = get_vae_model(M, beta, device)
    # LOAD WEIGHTS
    vae_model.load_state_dict(torch.load('beta_vae.pt', map_location=device))
    vae_model.eval() # Set to evaluation mode
    for param in vae_model.parameters():
        param.requires_grad = False # Freeze weights
    print("Successfully loaded and frozen beta_vae.pt")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),
        transforms.Lambda(lambda x: (x-0.5)*2.0),
        transforms.Lambda(lambda x: x.flatten())
    ])
    train_data = datasets.MNIST('data/', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Input dimension is M (32) because we are diffusing in latent space
    num_hidden = 128
    T = 1000
    network = FcNetwork(input_dim=M, num_hidden=num_hidden)
    latent_ddpm = DDPM(network, T=T).to(device)
    
    optimizer = torch.optim.Adam(latent_ddpm.parameters(), lr=1e-5)

    train_latent_ddpm(vae_model, latent_ddpm, optimizer, train_loader, ddpm_epochs, device)
    
    # Save the trained Latent DDPM
    torch.save(latent_ddpm.state_dict(), 'latent_ddpm.pt')


    latent_ddpm.eval()
    with torch.no_grad():
        print("Generating 4 samples...")
        # Sample z from the DDPM
        sampled_z = latent_ddpm.sample((4, M)).to(device)
        
        # Decode z back to images using the frozen VAE
        generated_images = vae_model.decoder(sampled_z).mean 
        
        # Reshape from flat (784) back to 2D image (1, 28, 28)
        generated_images = generated_images.view(4, 1, 28, 28)
        
        # Transform out of [-1, 1]
        generated_images = (generated_images / 2) + 0.5
        

        save_image(generated_images, 'latent_ddpm_samples.png', nrow=4)
        print("Saved samples to latent_ddpm_samples.png")