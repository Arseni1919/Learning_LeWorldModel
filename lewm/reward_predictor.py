import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from lewm.encoder import Encoder


class RewardPredictor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2 + action_dim + 1, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, z_prev: torch.Tensor, z: torch.Tensor,
                action: torch.Tensor, terminated: torch.Tensor) -> torch.Tensor:
        a_onehot = F.one_hot(action, num_classes=4).float()
        x = torch.cat([z_prev, z, a_onehot, terminated.unsqueeze(-1)], dim=-1)
        return self.net(x).squeeze(-1)


if __name__ == "__main__":
    OBS_DIM = 8
    ACTION_DIM = 4
    LATENT_DIM = 16

    env = gym.make("LunarLander-v3")
    encoder = Encoder(OBS_DIM, LATENT_DIM)
    reward_predictor = RewardPredictor(LATENT_DIM, ACTION_DIM)
    encoder.eval()
    reward_predictor.eval()

    obs, _ = env.reset()
    action = env.action_space.sample()
    next_obs, _, terminated, _, _ = env.step(action)

    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
    action_t = torch.tensor([action])
    terminated_t = torch.tensor([float(terminated)])

    with torch.no_grad():
        z_prev = encoder(obs_t)
        z = encoder(next_obs_t)
        r_hat = reward_predictor(z_prev, z, action_t, terminated_t)

    print(f"z_prev shape:      {z_prev.shape}")
    print(f"z shape:           {z.shape}")
    print(f"predicted reward:  {r_hat.item():.4f}")
