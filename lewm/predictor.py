import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from lewm.encoder import Encoder


class Predictor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, dropout: float = 0.1):
        super().__init__()
        self.action_dim = action_dim
        self.adaLN = nn.Linear(action_dim, 2 * latent_dim)
        nn.init.zeros_(self.adaLN.weight)
        nn.init.zeros_(self.adaLN.bias)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        a = F.one_hot(action, num_classes=self.action_dim).float()
        gamma, beta = self.adaLN(a).chunk(2, dim=-1)
        x = gamma * z + beta
        return self.mlp(x)


if __name__ == "__main__":
    OBS_DIM = 8
    ACTION_DIM = 4
    LATENT_DIM = 16

    env = gym.make("LunarLander-v3")
    encoder = Encoder(OBS_DIM, LATENT_DIM)
    predictor = Predictor(LATENT_DIM, ACTION_DIM)
    encoder.eval()
    predictor.eval()

    obs, _ = env.reset()
    action = env.action_space.sample()

    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    action_tensor = torch.tensor([action])

    with torch.no_grad():
        z = encoder(obs_tensor)
        z_next = predictor(z, action_tensor)

    print(f"latent shape:          {z.shape}")
    print(f"action:                {action}")
    print(f"predicted next latent: {z_next}")
