import torch
import torch.nn as nn


class AdaLN(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return (1 + gamma) * self.norm(x) + beta


class PredictorConcat(nn.Module):
    def __init__(self, latent_dim: int, n_classes: int = 3, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + n_classes, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )

    def forward(self, z, one_hot):
        return self.net(torch.cat([z, one_hot], dim=1))


class Predictor(nn.Module):
    def __init__(self, latent_dim: int, n_classes: int = 3, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden)
        self.adaLN1 = AdaLN(hidden, n_classes)
        self.fc2 = nn.Linear(hidden, latent_dim)
        self.adaLN2 = AdaLN(latent_dim, n_classes)

    def forward(self, z, one_hot):
        x = self.adaLN1(self.fc1(z), one_hot)
        x = torch.relu(x)
        x = self.adaLN2(self.fc2(x), one_hot)
        return x
