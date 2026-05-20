import torch
import torch.nn as nn
import gymnasium as gym


class Encoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
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

    encoder = Encoder(obs_dim, latent_dim)
    encoder.eval()

    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        z = encoder(obs_t)

    print(f"obs shape:     {obs_t.shape}")
    print(f"latent shape:  {z.shape}")
    print(f"latent vector: {z}")
