## About the VAE

The command to train is (assuming running from root of the repo, Windows):

```bash
python .\mini3\vae_node_latents.py train
```
and to sample
```bash
python .\mini3\vae_node_latents.py sample 5
```
For the visualizer, after having sampled
```bash
python .\mini3\graph_visualizer.py .\generated_graphs_vae.pt
```