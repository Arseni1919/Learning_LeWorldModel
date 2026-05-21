import torch
import gymnasium as gym
import numpy as np
from lewm.encoder import Encoder
from lewm.utils import collect_data, signed_log


OBS_DIM = 8
LATENT_DIM = 16
N_SAMPLES = 2000

device = torch.device("cpu")
encoder = Encoder(OBS_DIM * 2 + 1, LATENT_DIM)
ckpt = torch.load("data/checkpoint_final.pt", map_location=device)
encoder.load_state_dict(ckpt["encoder"])
encoder.eval()

env = gym.make("LunarLander-v3")
data = collect_data(env, N_SAMPLES)

prev_obs = torch.tensor(np.array([s[0] for s in data]), dtype=torch.float32)
obs = torch.tensor(np.array([s[1] for s in data]), dtype=torch.float32)
prev_r = signed_log(torch.tensor([s[2] for s in data], dtype=torch.float32)).unsqueeze(-1)
enc_in = torch.cat([prev_obs, obs, prev_r], dim=-1)

with torch.no_grad():
    Z = encoder(enc_in)

var = Z.var(dim=0)
dists = torch.cdist(Z[:500], Z[:500])
mean_dist = dists[dists > 0].mean()

print(f"per-dim variance:  {var.numpy().round(3)}")
print(f"mean variance:     {var.mean():.4f}")
print(f"dead dims (<0.01): {(var < 0.01).sum().item()} / {LATENT_DIM}")
print(f"mean pairwise L2:  {mean_dist:.4f}  (expected ~{(2 * LATENT_DIM) ** 0.5:.1f})")
