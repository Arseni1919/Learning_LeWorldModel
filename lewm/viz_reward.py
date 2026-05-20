import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gymnasium as gym
from lewm.encoder import Encoder
from lewm.predictor import Predictor
from lewm.reward_predictor import RewardPredictor
from lewm.planning import cem, a_star
from lewm.utils import signed_log
from tqdm import tqdm


OBS_DIM = 8
LATENT_DIM = 16
ACTION_DIM = 4
N_STEPS = 1000
BOUNDS = (-2.5, 2.5)

parser = argparse.ArgumentParser()
parser.add_argument("--planner", choices=["random", "cem", "a_star"], default="a_star")
args = parser.parse_args()

device = torch.device("cpu")

encoder = Encoder(OBS_DIM, LATENT_DIM)
predictor = Predictor(LATENT_DIM, ACTION_DIM)
reward_predictor = RewardPredictor(LATENT_DIM, ACTION_DIM)
ckpt = torch.load("data/checkpoint_final.pt", map_location=device)
encoder.load_state_dict(ckpt["encoder"])
predictor.load_state_dict(ckpt["predictor"])
reward_predictor.load_state_dict(
    torch.load("data/reward_predictor_final.pt", map_location=device)
)
encoder.eval()
predictor.eval()
reward_predictor.eval()

env = gym.make("LunarLander-v3")

if args.planner == "cem":
    def get_action(obs):
        with torch.no_grad():
            return cem(torch.tensor(obs, dtype=torch.float32), encoder, predictor, reward_predictor)
elif args.planner == "a_star":
    def get_action(obs):
        with torch.no_grad():
            return a_star(torch.tensor(obs, dtype=torch.float32), encoder, predictor, reward_predictor)
else:
    get_action = lambda obs: env.action_space.sample()

xs, ys, real_rewards, pred_rewards = [], [], [], []

while len(xs) < N_STEPS:
    obs, _ = env.reset()
    terminated = truncated = False
    while not (terminated or truncated):
        print(f"\rstep {len(xs):4d} | obs: {obs[:]} | reward: {real_rewards[-1] if real_rewards else 0:.3f} | pred: {pred_rewards[-1] if pred_rewards else 0:.3f}", end="")
        action = get_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
        action_t = torch.tensor([action])
        terminated_t = torch.tensor([float(terminated)])
        with torch.no_grad():
            z_prev = encoder(obs_t)
            z = encoder(next_obs_t)
            pred = reward_predictor(z_prev, z, action_t, terminated_t).item()
        xs.append(obs[0])
        ys.append(obs[1])
        real_rewards.append(reward)
        pred_rewards.append(pred)
        obs = next_obs
        if len(xs) >= N_STEPS:
            break

xs = np.array(xs)
ys = np.array(ys)
real_rewards = np.array(real_rewards)
pred_rewards = np.array(pred_rewards)

print(f"real  | mean: {real_rewards.mean():.3f}  std: {real_rewards.std():.3f}  "
      f"min: {real_rewards.min():.3f}  max: {real_rewards.max():.3f}")
print(f"pred  | mean: {pred_rewards.mean():.3f}  std: {pred_rewards.std():.3f}  "
      f"min: {pred_rewards.min():.3f}  max: {pred_rewards.max():.3f}")
print(f"corr  | {np.corrcoef(real_rewards, pred_rewards)[0, 1]:.3f}")

real_rewards = signed_log(real_rewards)

vmax = max(abs(real_rewards).max(), abs(pred_rewards).max())
norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
cmap = "RdYlGn"

fig, (ax_real, ax_pred) = plt.subplots(1, 2, figsize=(12, 5))
for ax, rewards, title in [
    (ax_real, real_rewards, "Real reward"),
    (ax_pred, pred_rewards, "Predicted reward"),
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
fig.colorbar(sc, cax=cbar_ax, label="reward")
plt.show()
