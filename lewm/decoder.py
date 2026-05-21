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
            nn.Linear(64, obs_dim + 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


if __name__ == "__main__":
    OBS_DIM = 8
    LATENT_DIM = 16

    env = gym.make("LunarLander-v3")
    obs, _ = env.reset()

    encoder = Encoder(OBS_DIM + 1, LATENT_DIM)
    decoder = Decoder(LATENT_DIM, OBS_DIM)
    encoder.eval()
    decoder.eval()

    enc_in = torch.cat([torch.tensor(obs, dtype=torch.float32), torch.tensor([0.0])]).unsqueeze(0)
    with torch.no_grad():
        z = encoder(enc_in)
        out = decoder(z)

    print(f"encoder input shape: {enc_in.shape}")
    print(f"latent shape:        {z.shape}")
    print(f"decoder output shape:{out.shape}  (obs_dim + 1 = {OBS_DIM + 1})")
    print(f"obs_hat:             {out.squeeze()[:OBS_DIM].numpy()}")
    print(f"reward_log_hat:      {out.squeeze()[-1].item():.4f}")
