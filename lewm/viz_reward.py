import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gymnasium as gym
from lewm.encoder import Encoder
from lewm.decoder import Decoder
from lewm.predictor import Predictor
from lewm.planning import cem, a_star
from lewm.utils import signed_log
from lewm.params import OBS_DIM, ACTION_DIM, LATENT_DIM
from tqdm import tqdm
N_RUNS = 10
BOUNDS = (-2.5, 2.5)

parser = argparse.ArgumentParser()
parser.add_argument("--planner", choices=["random", "cem", "a_star"], default="random")
args = parser.parse_args()

device = torch.device("cpu")

encoder = Encoder(OBS_DIM * 2 + 1, LATENT_DIM)
predictor = Predictor(LATENT_DIM, ACTION_DIM)
decoder = Decoder(LATENT_DIM, OBS_DIM)
ckpt = torch.load("data/checkpoint_final.pt", map_location=device)
encoder.load_state_dict(ckpt["encoder"])
predictor.load_state_dict(ckpt["predictor"])
decoder.load_state_dict(torch.load("data/decoder_final.pt", map_location=device))
encoder.eval()
predictor.eval()
decoder.eval()

env = gym.make("LunarLander-v3")

if args.planner == "cem":
    def get_action(obs, prev_obs, prev_reward_log):
        with torch.no_grad():
            return cem(torch.tensor(obs, dtype=torch.float32), encoder, predictor, decoder,
                       prev_obs=torch.tensor(prev_obs, dtype=torch.float32),
                       prev_reward_log=prev_reward_log)
elif args.planner == "a_star":
    def get_action(obs, prev_obs, prev_reward_log):
        with torch.no_grad():
            return a_star(torch.tensor(obs, dtype=torch.float32), encoder, predictor, decoder,
                          prev_obs=torch.tensor(prev_obs, dtype=torch.float32),
                          prev_reward_log=prev_reward_log)
else:
    def get_action(obs, prev_obs, prev_reward_log):
        return env.action_space.sample()

xs, ys, real_rewards, pred_rewards = [], [], [], []

for run in tqdm(range(N_RUNS), desc="Runs"):
    obs, _ = env.reset()
    prev_obs = obs.copy()
    prev_reward_log = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        print(f"\rstep {len(xs):4d} | obs: {obs[:]} | reward: "
              f"{real_rewards[-1] if real_rewards else 0:.3f} | pred: "
              f"{pred_rewards[-1] if pred_rewards else 0:.3f}", end="")
        action = get_action(obs, prev_obs, prev_reward_log)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        prev_obs_t = torch.tensor(prev_obs, dtype=torch.float32).unsqueeze(0)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        prev_r_t = torch.tensor([[prev_reward_log]], dtype=torch.float32)
        enc_in = torch.cat([prev_obs_t, obs_t, prev_r_t], dim=-1)
        action_t = torch.tensor([action])
        with torch.no_grad():
            z = encoder(enc_in)
            z_next = predictor(z, action_t)
            pred = decoder(z_next)[0, -1].item()
        xs.append(obs[0])
        ys.append(obs[1])
        real_rewards.append(reward)
        pred_rewards.append(pred)
        prev_obs = obs.copy()
        prev_reward_log = signed_log(reward)
        obs = next_obs

xs = np.array(xs)
ys = np.array(ys)
real_rewards = np.array(real_rewards)
pred_rewards = np.array(pred_rewards)

real_rewards_log = signed_log(real_rewards)

print(f"real  | mean: {real_rewards.mean():.3f}  std: {real_rewards.std():.3f}  "
      f"min: {real_rewards.min():.3f}  max: {real_rewards.max():.3f}")
print(f"pred  | mean: {pred_rewards.mean():.3f}  std: {pred_rewards.std():.3f}  "
      f"min: {pred_rewards.min():.3f}  max: {pred_rewards.max():.3f}")
print(f"corr  | {np.corrcoef(real_rewards_log, pred_rewards)[0, 1]:.3f}")

vmax = max(abs(real_rewards_log).max(), abs(pred_rewards).max())
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap = "RdYlGn"

fig, (ax_real, ax_pred) = plt.subplots(1, 2, figsize=(12, 5))
for ax, rewards, title in [
    (ax_real, real_rewards_log, "Real reward (log scale)"),
    (ax_pred, pred_rewards, "Predicted reward (log scale)"),
]:
    ax.set_xlim(*BOUNDS)
    ax.set_ylim(*BOUNDS)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.scatter([0], [0], color="black", marker="x", s=100, zorder=4)
    sc = ax.scatter(xs, ys, c=rewards, cmap=cmap, norm=norm, alpha=0.3, s=20)
    ax.set_title(title)

plt.subplots_adjust(right=0.88, wspace=0.3)
cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
fig.colorbar(sc, cax=cbar_ax, label="reward (log scale)")
plt.show()
