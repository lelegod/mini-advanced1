import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split

def calculate_graph_stats(pyg_graphs):
    """Calculates node degrees, clustering coefficients, and eigenvector centralities."""
    degrees, clusterings, eigenvector_cents = [], [], []
    
    for data in pyg_graphs:
        G = to_networkx(data, to_undirected=True)
        if G.number_of_nodes() == 0:
            continue
            
        degrees.extend([d for _, d in G.degree()])
        clusterings.extend(list(nx.clustering(G).values()))
        
        # Eigenvector centrality
        if G.number_of_edges() > 0:
            try:
                ec = nx.eigenvector_centrality_numpy(G)
                eigenvector_cents.extend(list(ec.values()))
            except Exception:
                eigenvector_cents.extend([0.0] * G.number_of_nodes())
        else:
            eigenvector_cents.extend([0.0] * G.number_of_nodes())
            
    return np.array(degrees), np.array(clusterings), np.array(eigenvector_cents)

def plot_histogram_row(axes, row_idx, train_data, er_data, vae_data, metric_name, color):
    # Compute shared bins across all three distributions for this specific metric
    combined_data = np.concatenate([train_data, er_data, vae_data])
    bins = np.histogram_bin_edges(combined_data, bins=30)
    
    datasets = [train_data, er_data, vae_data]
    titles = [f"Train: {metric_name}", f"ER: {metric_name}", f"VAE: {metric_name}"]
    
    for col_idx in range(3):
        ax = axes[row_idx, col_idx]
        ax.hist(datasets[col_idx], bins=bins, color=color, alpha=0.7, edgecolor='black', density=True)
        ax.set_title(titles[col_idx], fontsize=14)
        ax.set_xlabel("Value", fontsize=20)
        ax.set_ylabel("Density", fontsize=20)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(axis='y', alpha=0.3)

def main():
    # Load standardized training split
    dataset = TUDataset(root="./data/", name="MUTAG")
    rng = torch.Generator().manual_seed(0)
    train_dataset, _, _ = random_split(dataset, (100, 44, 44), generator=rng)
    train_graphs = [g for g in train_dataset]

    # Load generated samples
    try:
        er_graphs = torch.load("mini3/generated_graphs_erdos_renyi.pt", weights_only=False)
        vae_graphs = torch.load("mini3/generated_graphs_vae.pt", weights_only=False)
    except FileNotFoundError as e:
        print(f"File loading error: {e}")
        return

    
    train_deg, train_clu, train_eig = calculate_graph_stats(train_graphs)
    er_deg, er_clu, er_eig = calculate_graph_stats(er_graphs)
    vae_deg, vae_clu, vae_eig = calculate_graph_stats(vae_graphs)

    # Initialize 3x3 Plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), constrained_layout=True)
    fig.suptitle("Graph Statistics: Empirical vs Generated Distributions", fontsize=30, fontweight='bold')

    # Plot Rows
    plot_histogram_row(axes, 0, train_deg, er_deg, vae_deg, "Node Degree", "skyblue")
    plot_histogram_row(axes, 1, train_clu, er_clu, vae_clu, "Clustering Coeff", "lightcoral")
    plot_histogram_row(axes, 2, train_eig, er_eig, vae_eig, "Eigenvector Cent.", "lightgreen")

    # Save and Show
    plt.savefig("mini3/graph_statistics_histograms.png", dpi=300)
    print("Plot saved successfully as 'graph_statistics_histograms.png'.")

if __name__ == "__main__":
    main()