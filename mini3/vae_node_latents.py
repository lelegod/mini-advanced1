import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import TUDataset

LOGITS_DENOMINATOR = 1

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        # Use LayerNorm to keep the scale of node features distinct
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.conv_mu = GCNConv(hidden_dim, latent_dim)
        self.conv_logstd = GCNConv(hidden_dim, latent_dim)

    def forward(self, x, edge_index):
        # 1. First convolution
        h = self.conv1(x, edge_index)
        h = self.norm(h)
        h = F.leaky_relu(h) # LeakyRelu helps prevent dead neurons
        
        # 2. Output layers
        mu = self.conv_mu(h, edge_index)
        logstd = self.conv_logstd(h, edge_index)
        
        return mu, logstd


class NodeVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim)
        self.bias = torch.nn.Parameter(torch.tensor([-3.0])) # Start at ~Sigmoid(-3) = 0.04

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return (z @ z.t()) / LOGITS_DENOMINATOR


import numpy as np
from torch_geometric.data import Data
from collections import defaultdict

class NodeLatentVAEGenerator:
    def __init__(self, train_dataset, model):
        self.train_dataset = train_dataset
        self.model = model

    def sample_graph_from_posterior(self, g):
        x = g.x
        edge_index = g.edge_index

        mu, logstd = self.model.encoder(x, edge_index)
        z = self.model.reparameterize(mu, logstd)

        logits = self.model.decode(z) + self.model.bias
        prob_adj = torch.sigmoid(logits)

        prob_adj.fill_diagonal_(0)

        adj = (prob_adj > 0.5).float()
        edge_index_new = adj.nonzero(as_tuple=False).t().contiguous()

        return Data(edge_index=edge_index_new, num_nodes=g.num_nodes)


    def forward(self, num_samples=1):
        graphs = []
        for _ in range(num_samples):
            idx = torch.randint(0, len(self.train_dataset), (1,)).item()
            g = self.train_dataset[idx]
            graphs.append(self.sample_graph_from_posterior(g))
        return graphs



def train(model_path):
    dataset = TUDataset(root="./data/", name="MUTAG")

    rng = torch.Generator().manual_seed(0)
    train_dataset, _, _ = random_split(dataset, (100, 44, 44), generator=rng)

    model = NodeVAE(
        in_channels=dataset.num_features,
        hidden_dim=32,
        latent_dim=16
    )

    train_vae(train_dataset, model)

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

from torch_geometric.loader import DataLoader

def train_vae(train_dataset, model, epochs=50):
    # Use a DataLoader for stability
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(epochs):
        total_loss = 0
        
        for data in train_loader:
            optimizer.zero_grad()

            x, edge_index, batch = data.x, data.edge_index, data.batch
            mu, logstd = model.encoder(x, edge_index)
            z = model.reparameterize(mu, logstd)

            # Reconstruction: Inner product decoder
            # 1. Compute logits
            logits = (z @ z.t()) / LOGITS_DENOMINATOR + model.bias

            # 2. Build correct adjacency (block-diagonal)
            adj_dense = torch.zeros_like(logits)
            adj_dense[edge_index[0], edge_index[1]] = 1.0

            # 3. Mask: same graph, no self-loops
            same_graph_mask = (batch.unsqueeze(0) == batch.unsqueeze(1))
            no_self_loop_mask = ~torch.eye(data.num_nodes, device=z.device).bool()
            final_mask = same_graph_mask & no_self_loop_mask

            # 4. Class imbalance weighting
            valid_adj = adj_dense[final_mask]
            num_pos = valid_adj.sum()
            num_neg = valid_adj.numel() - num_pos
            pos_weight = num_neg / (num_pos + 1e-6)

            # 5. BCE loss
            recon_loss = F.binary_cross_entropy_with_logits(
                logits[final_mask],
                adj_dense[final_mask],
                pos_weight=torch.tensor([pos_weight], device=z.device)
            )
            
            """ # Get ground truth adjacency
            adj = to_dense_adj(edge_index, max_num_nodes=data.num_nodes).squeeze(0)

            # Create mask for all elements EXCEPT the diagonal
            mask = ~torch.eye(data.num_nodes, device=z.device).bool()
            #print(mask)

            pos_weight = (adj.numel() - adj.sum()) / (adj.sum())

            # Only apply loss to the actual edges (off-diagonal)
            recon_loss = F.binary_cross_entropy_with_logits(
                logits[mask], 
                adj[mask],
                pos_weight=pos_weight
            ) """
            #print(logits)
            #print(logits[mask])
            
            # Stabilized Loss
            #recon_loss = F.binary_cross_entropy_with_logits(logits, adj)

            # Refined KL Divergence
            # Formula: 0.5 * sum(exp(2*logstd) + mu^2 - 1 - 2*logstd)
            kl = -0.5 * torch.mean(torch.sum(1 + 2*logstd - mu**2 - torch.exp(2*logstd), dim=1))
            
            # Norm factor: helps scale KL relative to the number of nodes
            # A common trick in VGAE
            kl = kl / data.num_nodes 

            # Warm-up schedule: reach 1.0 faster (e.g., by epoch 20)
            beta = min(1.0, epoch / 5) 
            
            loss = recon_loss + beta * kl
            loss.backward()
            #print(f"Encoder Grad: {model.encoder.conv1.lin.weight.grad.abs().mean().item():.8f}")
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:03d} | Avg Loss: {total_loss / len(train_loader):.4f}, recon: {recon_loss:.2f}, beta*kl: {beta*kl:.2f}")

def sample(num_samples, model_path):
    dataset = TUDataset(root="./data/", name="MUTAG")

    model = NodeVAE(
        in_channels=dataset.num_features,
        hidden_dim=32,
        latent_dim=16
    )

    model.load_state_dict(torch.load(model_path))
    model.eval()

    rng = torch.Generator().manual_seed(0)
    train_dataset, _, _ = random_split(dataset, (100, 44, 44), generator=rng)

    generator = NodeLatentVAEGenerator(train_dataset, model)

    graphs = generator.forward(num_samples)

    torch.save(graphs, ".\\mini3\\generated_graphs_vae.pt")

    print(f"Generated {len(graphs)} graphs.")

def main():
    parser = argparse.ArgumentParser()
    model_path = ".\\mini3\\vae_node_latent.pt"

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    subparsers.add_parser("train")

    # Sample command
    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("num_samples", type=int)

    args = parser.parse_args()

    if args.command == "train":
        train(model_path)

    elif args.command == "sample":
        sample(args.num_samples, model_path)


if __name__ == "__main__":
    main()