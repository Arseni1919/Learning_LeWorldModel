import torch
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from lewm.encoder import Encoder
from lewm.train_jepa import collect_data


OBS_DIM = 8
LATENT_DIM = 16
N_SAMPLES = 2000
M = 10

device = torch.device("cpu")
encoder = Encoder(OBS_DIM, LATENT_DIM)
ckpt = torch.load("data/checkpoint_final.pt", map_location=device)
encoder.load_state_dict(ckpt["encoder"])
encoder.eval()

env = gym.make("LunarLander-v3")
data = collect_data(env, N_SAMPLES)

obs = torch.tensor(np.array([s[1] for s in data]), dtype=torch.float32)

with torch.no_grad():
    Z = encoder(obs)

torch.manual_seed(0)
U = torch.nn.functional.normalize(torch.randn(LATENT_DIM, M), dim=0)
H = (Z @ U).numpy()

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
    ax.hist(H[:, i], bins=40, density=True, color="steelblue", alpha=0.7)
    x = np.linspace(-3, 3, 200)
    ax.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color="red", lw=1.5)
    ax.set_xlim(-3, 3)
    ax.set_title(f"projection {i + 1}")

plt.tight_layout()
plt.show()
input("Press Enter to exit...")
