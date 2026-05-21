import numpy as np
import torch
from typing import Callable
from tqdm import tqdm


def collect_data(env, n_steps: int, sample_action_fn: Callable = None) -> list[tuple]:
    data = []
    obs, _ = env.reset()
    prev_obs = obs.copy()
    prev_reward = 0.0
    for _ in tqdm(range(n_steps), desc="collecting", unit="step"):
        action = sample_action_fn(obs) if sample_action_fn else env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        data.append((prev_obs.copy(), obs.copy(), prev_reward, action, next_obs.copy(), reward, terminated))
        prev_obs = obs
        prev_reward = reward
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
            prev_obs = obs.copy()
            prev_reward = 0.0
    return data


def signed_log(x):
    if isinstance(x, torch.Tensor):
        return x.sign() * torch.log1p(x.abs())
    return np.sign(x) * np.log1p(np.abs(x))


def signed_exp(x):
    if isinstance(x, torch.Tensor):
        return x.sign() * (torch.expm1(x.abs()))
    return np.sign(x) * np.expm1(np.abs(x))
