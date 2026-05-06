import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np

# Constants for MUTAG dataset
MAX_NODES = 28  # Maximum nodes in a MUTAG graph
EDGE_PROBABILITY_CUTOFF = 0.5

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Graph-level latent heads
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logstd = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, edge_index, batch):
        # 1. Node embeddings
        h = self.conv1(x, edge_index)
        h = F.leaky_relu(self.norm(h))
        h = self.conv2(h, edge_index)
        h = F.leaky_relu(h)
        
        # 2. Global Pooling (Node-level -> Graph-level)
        hg = global_mean_pool(h, batch) # Result: [batch_size, hidden_dim]
        
        # 3. Output graph latents
        mu = self.fc_mu(hg)
        logstd = self.fc_logstd(hg)
        
        return mu, logstd

class GraphVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, max_nodes):
        super().__init__()
        self.max_nodes = max_nodes
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim)
        
        # Decoder: Maps graph latent back to a flattened adjacency matrix
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, max_nodes * max_nodes)
        )

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Output flattened logits and reshape to (batch, N, N)
        logits = self.decoder(z)
        return logits.view(-1, self.max_nodes, self.max_nodes)

class GraphLatentVAEGenerator:
    def __init__(self, model):
        self.model = model

    def forward(self, num_samples=1):
        self.model.eval()
        graphs = []
        with torch.no_grad():
            # Sample from the prior N(0, I)
            z = torch.randn(num_samples, self.model.encoder.fc_mu.out_features)
            logits = self.model.decode(z)
            prob_adj = torch.sigmoid(logits)

            for i in range(num_samples):
                adj = prob_adj[i]
                # Symmetrize and remove self-loops
                adj = (adj + adj.t()) / 2
                adj.fill_diagonal_(0)
                print(np.round(adj,3))
                # Apply cutoff
                bin_adj = (adj > EDGE_PROBABILITY_CUTOFF).float()
                edge_index = bin_adj.nonzero(as_tuple=False).t().contiguous()
                
                graphs.append(Data(edge_index=edge_index, num_nodes=self.model.max_nodes))
        return graphs

def train_vae(train_dataset, model, epochs=200):
    loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in loader:
            opt.zero_grad()
            
            # Encode
            mu, logstd = model.encoder(data.x, data.edge_index, data.batch)
            z = model.reparameterize(mu, logstd)
            
            # Decode (Reconstruct Adjacency)
            logits = model.decode(z)
            
            # Prepare Target: Dense Adjacency with Padding
            # to_dense_adj returns [batch, max_nodes_in_batch, max_nodes_in_batch]
            adj_dense = to_dense_adj(data.edge_index, data.batch)
            
            # Pad target to MAX_NODES
            batch_size = adj_dense.size(0)
            target_adj = torch.zeros(batch_size, MAX_NODES, MAX_NODES).to(adj_dense.device)
            curr_nodes = adj_dense.size(1)
            target_adj[:, :curr_nodes, :curr_nodes] = adj_dense

            # Loss: Reconstruction (BCE) + KL
            recon_loss = F.binary_cross_entropy_with_logits(logits, target_adj)
            kl = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - torch.exp(2 * logstd), dim=1))
            
            beta = min(0.1, epoch / 100)
            loss = recon_loss + beta * kl
            
            loss.backward()
            opt.step()
            total_loss += loss.item()
            avg_recon += recon_loss.item()
            avg_kl += kl.item()
            avg_loss += loss.item()

        if epoch % 10 == 0:
            n = len(loader)
            # More informative loss printing[cite: 1]
            print(f"Epoch {epoch:03d} | Total: {avg_loss/n:.4f} | Recon: {avg_recon/n:.4f} | KL: {avg_kl/n:.4f} | Beta: {beta:.3f}")

def train(model_path):
    dataset = TUDataset(root="./data/", name="MUTAG")
    rng = torch.Generator().manual_seed(0)
    train_dataset, _, _ = random_split(dataset, (100, 44, 44), generator=rng)

    model = GraphVAE(
        in_channels=dataset.num_features,
        hidden_dim=64,
        latent_dim=32,
        max_nodes=MAX_NODES
    )

    train_vae(train_dataset, model)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

def sample(num_samples, model_path):
    dataset = TUDataset(root="./data/", name="MUTAG")
    model = GraphVAE(
        in_channels=dataset.num_features,
        hidden_dim=64,
        latent_dim=32,
        max_nodes=MAX_NODES
    )

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    model.load_state_dict(torch.load(model_path))
    generator = GraphLatentVAEGenerator(model)
    graphs = generator.forward(num_samples)

    output_path = os.path.join("mini3", "generated_graphs_vae.pt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(graphs, output_path)
    print(f"Generated {len(graphs)} graphs and saved to {output_path}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vae_graph_latent.pt", help="Path to model weights.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train")
    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("num_samples", type=int)

    args = parser.parse_args()
    if args.command == "train":
        train(args.model)
    elif args.command == "sample":
        sample(args.num_samples, args.model)

if __name__ == "__main__":
    main()