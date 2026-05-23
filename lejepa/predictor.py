import torch
import torch.nn as nn


class Predictor(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )

    def forward(self, z, one_hot):
        return self.net(torch.cat([z, one_hot], dim=1))
