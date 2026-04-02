"""
Plot latent-space geodesics for a single-decoder VAE (Part A) and
an ensemble-decoder VAE (Part B) side by side.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from ensemble_vae import (
    VAE,
    GaussianDecoder,
    GaussianEncoder,
    GaussianPrior,
    compute_geodesic,
    compute_geodesic_ensemble,
    new_decoder,
    new_encoder,
)
from torchvision import datasets, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
M = 2  # latent dim
NUM_PAIRS = 25
NUM_T = 20
NUM_STEPS = 500
LR = 1e-2
SEED = 42
NUM_CLASSES = 3
NUM_TEST_DATA = 2048
PART_A_FOLDER = os.path.join(BASE_DIR, "experiment_partA")
PART_B_FOLDER = os.path.join(BASE_DIR, "partB")
NUM_DECODERS = 10  # decoder_0..decoder_9 in partB
MC_SAMPLES = 4


def load_model(model_path):
    model = VAE(
        GaussianPrior(M),
        GaussianDecoder(new_decoder()),
        GaussianEncoder(new_encoder()),
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def get_test_loader(batch_size=64):
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    idx = test_tensors.targets < NUM_CLASSES
    data = test_tensors.data[idx][:NUM_TEST_DATA].unsqueeze(1).float() / 255
    targets = test_tensors.targets[idx][:NUM_TEST_DATA]
    ds = torch.utils.data.TensorDataset(data, targets)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


def compute_latent_points(model, loader):
    all_z, all_y = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            z_mean = model.encoder(x).mean
            all_z.append(z_mean.cpu())
            all_y.append(y.cpu())
    return torch.cat(all_z, 0), torch.cat(all_y, 0)



# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def plot_on_ax(ax, all_z, all_y, geodesics, pairs, title):
    # Data points
    ax.scatter(
        all_z[:, 0].numpy(),
        all_z[:, 1].numpy(),
        c=all_y.numpy(),
        cmap="tab10",
        s=3,
        alpha=0.4,
        zorder=2,
    )

    # Geodesics and straight lines
    for curve, (a, b) in zip(geodesics, pairs):
        z_start = all_z[a].numpy()
        z_end = all_z[b].numpy()
        # Straight line
        ax.plot(
            [z_start[0], z_end[0]],
            [z_start[1], z_end[1]],
            color="blue",
            linewidth=1.0,
            linestyle="--",
            alpha=0.7,
            zorder=3,
        )
        # Pullback geodesic
        c_np = curve.numpy()
        ax.plot(
            c_np[:, 0], c_np[:, 1], color="orange", linewidth=1.5, alpha=0.85, zorder=4
        )

    # Legend
    ax.plot([], [], "o", color="orange", label="Pullback geodesic", markersize=5)
    ax.plot([], [], "o", color="blue", label="Straight line", markersize=5)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.8)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("z1")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    test_loader = get_test_loader()

    print("Loading Part B model (shared encoder) …")
    model_b = load_model(os.path.join(PART_B_FOLDER, "model.pt"))
    all_z, all_y = compute_latent_points(model_b, test_loader)

    print("Loading Part A decoder …")
    model_a = load_model(os.path.join(PART_A_FOLDER, "model.pt"))
    decoder_net_a = model_a.decoder.decoder_net.cpu()
    decoder_net_a.eval()

    print("Loading Part B ensemble decoders …")
    decoder_nets_b = []
    for i in range(NUM_DECODERS):
        dec = new_decoder()
        dec.load_state_dict(
            torch.load(
                os.path.join(PART_B_FOLDER, f"decoder_{i}.pt"), map_location="cpu"
            )
        )
        dec = dec.cpu().eval()
        decoder_nets_b.append(dec)

    # --- Select random pairs (same for both subplots) ---
    torch.manual_seed(SEED)
    idx = torch.randperm(len(all_z))[: NUM_PAIRS * 2]
    pairs = idx.reshape(NUM_PAIRS, 2)

    # --- Compute geodesics for Part A ---
    print(f"Computing {NUM_PAIRS} geodesics (Part A) …")
    geodesics_a = []
    for i, (a, b) in enumerate(tqdm(pairs, desc="Part A geodesics")):
        curve = compute_geodesic(
            all_z[a],
            all_z[b],
            decoder_net_a,
            num_t=NUM_T,
            lr=LR,
            num_steps=NUM_STEPS,
        )
        geodesics_a.append(curve.detach().cpu())

    # --- Compute geodesics for Part B ---
    print(f"Computing {NUM_PAIRS} ensemble geodesics (Part B) …")
    geodesics_b = []
    for i, (a, b) in enumerate(tqdm(pairs, desc="Part B geodesics")):
        curve = compute_geodesic_ensemble(
            all_z[a],
            all_z[b],
            decoder_nets_b,
            S=MC_SAMPLES,
            num_t=NUM_T,
            lr=LR,
            num_steps=NUM_STEPS,
        )
        geodesics_b.append(curve.detach().cpu())

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.supylabel("z2")

    plot_on_ax(
        ax1,
        all_z,
        all_y,
        geodesics_a,
        pairs,
        "Geodesics with single decoder (Part A)",
    )
    plot_on_ax(
        ax2,
        all_z,
        all_y,
        geodesics_b,
        pairs,
        "Geodesics with ensemble decoders (Part B)",
    )

    plt.tight_layout()
    out_path = "geodesics_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
