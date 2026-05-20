import torch
import torch.nn as nn
import gymnasium as gym
from lewm.encoder import Encoder


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, obs_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


if __name__ == "__main__":
    OBS_DIM = 8
    LATENT_DIM = 16

    env = gym.make("LunarLander-v3")
    obs, _ = env.reset()

    encoder = Encoder(OBS_DIM, LATENT_DIM)
    decoder = Decoder(LATENT_DIM, OBS_DIM)
    encoder.eval()
    decoder.eval()

    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        z = encoder(obs_t)
        obs_hat = decoder(z)

    print(f"latent shape:        {z.shape}")
    print(f"reconstructed shape: {obs_hat.shape}")
    print(f"original obs:        {obs_t.squeeze().numpy()}")
    print(f"reconstructed obs:   {obs_hat.squeeze().numpy()}")
