import torch
import matplotlib.pyplot as plt
import gymnasium as gym
from lewm.encoder import Encoder
from lewm.decoder import Decoder
from lewm.utils import signed_log, signed_exp
from lewm.params import OBS_DIM, LATENT_DIM

ARROW_SCALE = 0.2
MAX_SPEED = 10.0
BOUNDS = (-2.5, 2.5)

device = torch.device("cpu")

encoder = Encoder(OBS_DIM * 2 + 1, LATENT_DIM)
decoder = Decoder(LATENT_DIM, OBS_DIM)
ckpt = torch.load("data/checkpoint_final.pt", map_location=device)
encoder.load_state_dict(ckpt["encoder"])
decoder.load_state_dict(torch.load("data/decoder_final.pt", map_location=device))
encoder.eval()
decoder.eval()


def plot_state(ax, x, y, vx, vy, title):
    ax.cla()
    ax.set_xlim(*BOUNDS)
    ax.set_ylim(*BOUNDS)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.scatter([0], [0], color="green", marker="x", s=100, zorder=4)
    speed = (abs(vx) + abs(vy)) / 2
    color = plt.cm.hot(speed / MAX_SPEED)
    ax.scatter(x, y, color="steelblue", s=80, zorder=5)
    ax.quiver(x, y, vx * ARROW_SCALE, vy * ARROW_SCALE,
              angles="xy", scale_units="xy", scale=1, color=color, width=0.02)
    ax.set_title(title)


env = gym.make("LunarLander-v3")
obs, _ = env.reset()
prev_obs = obs.copy()
prev_reward_log = 0.0

fig, (ax_real, ax_latent) = plt.subplots(1, 2, figsize=(12, 5))
plt.ion()

while True:
    prev_obs_t = torch.tensor(prev_obs, dtype=torch.float32).unsqueeze(0)
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    prev_r_t = torch.tensor([[prev_reward_log]], dtype=torch.float32)
    enc_in = torch.cat([prev_obs_t, obs_t, prev_r_t], dim=-1)
    with torch.no_grad():
        out = decoder(encoder(enc_in)).squeeze().numpy()
    obs_hat = out[:OBS_DIM]
    real_reward = signed_exp(prev_reward_log)
    pred_reward = signed_exp(out[-1])

    plot_state(ax_real, obs[0], obs[1], obs[2], obs[3],
               f"Real observation  |  reward: {real_reward:.3f}")
    plot_state(ax_latent, obs_hat[0], obs_hat[1], obs_hat[2], obs_hat[3],
               f"Decoded latent  |  pred reward: {pred_reward:.3f}")

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

    input("Press Enter for next step...")

    prev_obs = obs.copy()
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    prev_reward_log = signed_log(reward)

    if terminated or truncated:
        obs, _ = env.reset()
        prev_obs = obs.copy()
        prev_reward_log = 0.0
