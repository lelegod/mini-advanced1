import torch
import torch.utils.data
from torchvision import datasets, transforms
from tqdm import tqdm

from config import M, BETA_VAE, BATCH_SIZE, TRAIN_SIZE, DDPM_EPOCHS, DDPM_PATIENCE, NUM_HIDDEN, DDPM_LR
from beta_VAE import get_vae_model, EarlyStopping
from ddpm import DDPM, FcNetwork 

def train_latent_ddpm(vae, ddpm, optimizer, train_loader, val_loader, epochs, device, z_mean, z_std, patience=100):
    early_stopping = EarlyStopping(patience=patience, min_delta=0.01)

    for epoch in range(epochs):
        ddpm.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train DDPM]")

        for x, _ in progress_bar:
            x = x.to(device)
            
            with torch.no_grad():
                z = vae.encoder(x).sample() 
                z = (z - z_mean) / z_std
            
            optimizer.zero_grad()
            loss = ddpm.negative_elbo(z) 
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_train_loss = train_loss / len(train_loader)

        ddpm.eval()
        val_loss = 0
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                z = vae.encoder(x).sample()
                z = (z - z_mean) / z_std
                loss = ddpm.negative_elbo(z)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"End of Epoch {epoch+1} | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            break

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ddpm_epochs = DDPM_EPOCHS
    batch_size = BATCH_SIZE

    vae_model = get_vae_model(M, BETA_VAE, device)
    vae_model.load_state_dict(torch.load('beta_vae.pt', map_location=device))
    vae_model.eval()
    for param in vae_model.parameters():
        param.requires_grad = False
    print("Successfully loaded and frozen beta_vae.pt")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.rand(x.shape)/255),
        transforms.Lambda(lambda x: (x-0.5)*2.0),
        transforms.Lambda(lambda x: x.flatten())
    ])

    full_train_data = datasets.MNIST('data/', train=True, download=True, transform=transform)
    train_size = TRAIN_SIZE
    val_size = len(full_train_data) - train_size
    print(val_size)
    train_data, val_data = torch.utils.data.random_split(full_train_data, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    print("Calculating Latent Stats (Mean/Std)...")
    all_z = []
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            if i > 20: break 
            all_z.append(vae_model.encoder(x.to(device)).mean)
    all_z = torch.cat(all_z, dim=0)
    z_mean = all_z.mean(dim=0, keepdim=True)
    z_std = all_z.std(dim=0, keepdim=True) + 1e-5
    torch.save({'mean': z_mean, 'std': z_std}, 'latent_stats.pt')

    num_hidden = NUM_HIDDEN
    T = 1000
    network = FcNetwork(input_dim=M, num_hidden=num_hidden)
    latent_ddpm = DDPM(network, T=T).to(device)
    optimizer = torch.optim.Adam(latent_ddpm.parameters(), lr=DDPM_LR)
    train_latent_ddpm(vae_model, latent_ddpm, optimizer, train_loader, val_loader, 
                      ddpm_epochs, device, z_mean, z_std)
    
    torch.save(latent_ddpm.state_dict(), 'latent_ddpm.pt')
    print("Saved Latent DDPM weights to latent_ddpm.pt")