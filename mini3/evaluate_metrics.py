import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split

def get_graph_hash(pyg_graph):
    """Computes the Weisfeiler-Lehman hash for structural isomorphism checking."""
    G = to_networkx(pyg_graph, to_undirected=True)
    return nx.weisfeiler_lehman_graph_hash(G)

def calculate_metrics(train_graphs, generated_graphs):
    """Calculates Novelty, Uniqueness, and Novel+Unique percentages."""
    train_hashes = set(get_graph_hash(g) for g in train_graphs)
    gen_hashes = [get_graph_hash(g) for g in generated_graphs]
    
    total_gen = len(gen_hashes)
    if total_gen == 0:
        return 0.0, 0.0, 0.0

    unique_gen_hashes = set(gen_hashes)
    
    uniqueness = len(unique_gen_hashes) / total_gen
    novelty = sum(1 for h in gen_hashes if h not in train_hashes) / total_gen
    novel_unique = len(unique_gen_hashes - train_hashes) / total_gen

    return novelty * 100, uniqueness * 100, novel_unique * 100

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
    
    # Sanity Check
    hash1 = get_graph_hash(train_graphs[0])
    hash2 = get_graph_hash(train_graphs[0])
    print(f"Sanity Check - Same graph hashes match: {hash1 == hash2}")


    # Compute evaluation metrics
    er_nov, er_uni, er_nu = calculate_metrics(train_graphs, er_graphs)
    vae_nov, vae_uni, vae_nu = calculate_metrics(train_graphs, vae_graphs)

    # Output markdown table
    print("\n| Metric | Novel | Unique | Novel+Unique |")
    print(f"| Baseline (Erdős-Rényi) | {er_nov:.1f}% | {er_uni:.1f}% | {er_nu:.1f}% |")
    print(f"| Deep Generative Model (VAE) | {vae_nov:.1f}% | {vae_uni:.1f}% | {vae_nu:.1f}% |")

if __name__ == "__main__":
    main()