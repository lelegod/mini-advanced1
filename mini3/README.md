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


To run the graph level latent instead, the commands are

```bash
python .\mini3\vae_graph_latents.py train
```
and to sample
```bash
python .\mini3\vae_graph_latents.py sample 5
```
The generated graphs have the same name, so the other commands are the same