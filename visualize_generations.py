#!/usr/bin/env python3
"""
Script to visualize generated digit images from different models.
Creates a grid with models as rows and digits as columns.
"""

import os

import matplotlib.pyplot as plt
from PIL import Image

# Define the folders and models
models = ["flow", "ddpm", "latent_ddpm"]
digits = [3, 4, 5, 6]
base_path = "generations"

# Create figure with subplots
fig, axes = plt.subplots(
    len(models), len(digits), figsize=(len(digits) * 2, len(models) * 2)
)

# Load and display images
for i, model in enumerate(models):
    for j, digit in enumerate(digits):
        img_path = os.path.join(base_path, model, f"{digit}.png")

        # Load image
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("L")
            axes[i, j].imshow(img, cmap="gray", vmin=0, vmax=255)
        else:
            # If image doesn't exist, show blank
            axes[i, j].text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                transform=axes[i, j].transAxes,
            )

        # Remove axes completely
        axes[i, j].axis("off")


    # Add row labels (model names) on left side
    axes[i, 0].set_ylabel(
        model.upper().replace("_", " "), fontsize=12, rotation=90, labelpad=10
    )

# Remove all margins and add tight layout
plt.subplots_adjust(
    left=0.08, right=0.98, top=0.95, bottom=0.02, wspace=0.05, hspace=0.1
)

# Save the figure
output_path = os.path.join(base_path, "comparison.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved comparison plot to: {output_path}")
