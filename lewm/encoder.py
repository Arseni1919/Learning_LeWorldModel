import torch
import torch.nn as nn
import gymnasium as gym
from lewm.utils import signed_log


class Encoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


if __name__ == "__main__":
    env = gym.make("LunarLander-v3")
    obs, _ = env.reset()
    obs_dim = env.observation_space.shape[0]
    latent_dim = 16

    encoder = Encoder(obs_dim * 2 + 1, latent_dim)
    encoder.eval()

    prev_obs = obs.copy()
    prev_reward = 0.0
    enc_in = torch.cat([
        torch.tensor(prev_obs, dtype=torch.float32),
        torch.tensor(obs, dtype=torch.float32),
        torch.tensor([signed_log(prev_reward)], dtype=torch.float32),
    ]).unsqueeze(0)

    with torch.no_grad():
        z = encoder(enc_in)

    print(f"enc_in shape:  {enc_in.shape}")
    print(f"latent shape:  {z.shape}")
    print(f"latent vector: {z}")
