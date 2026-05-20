import time
import torch
import gymnasium as gym
from tqdm import tqdm
from lewm.encoder import Encoder
from lewm.predictor import Predictor
from lewm.reward_predictor import RewardPredictor


def cem(obs: torch.Tensor, encoder, predictor, reward_predictor,
        H: int = 10, N: int = 100, n_iter: int = 5,
        elite_frac: float = 0.1, smoothing: float = 0.1) -> int:
    device = obs.device
    ACTION_DIM = 4
    z0 = encoder(obs.unsqueeze(0)).expand(N, -1)
    n_elite = max(1, int(N * elite_frac))
    probs = torch.ones(H, ACTION_DIM, device=device) / ACTION_DIM
    actions = None
    for _ in range(n_iter):
        actions = torch.stack([
            torch.multinomial(probs[h], N, replacement=True) for h in range(H)
        ]).T  # (N, H)
        z = z0.clone()
        scores = torch.zeros(N, device=device)
        terminated = torch.zeros(N, device=device)
        for h in range(H):
            z_next = predictor(z, actions[:, h])
            scores += reward_predictor(z, z_next, actions[:, h], terminated)
            z = z_next
        elite = actions[scores.topk(n_elite).indices]  # (n_elite, H)
        counts = torch.zeros(H, ACTION_DIM, device=device)
        for h in range(H):
            counts[h].scatter_add_(0, elite[:, h], torch.ones(n_elite, device=device))
        probs = (1 - smoothing) * counts / n_elite + smoothing / ACTION_DIM
    return actions[scores.argmax(), 0].item()


def demo(encoder, predictor, reward_predictor, device):
    env = gym.make("LunarLander-v3", render_mode="human")
    obs, _ = env.reset()
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            action = cem(obs_tensor, encoder, predictor, reward_predictor)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
    env.close()
    time.sleep(1)
    print(f"  demo | total reward: {total_reward:.2f}")


def evaluate(env, encoder, predictor, reward_predictor, n_episodes: int, device) -> list[float]:
    rewards = []
    for _ in tqdm(range(n_episodes), desc="evaluating", unit="episode"):
        obs, _ = env.reset()
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                action = cem(obs_tensor, encoder, predictor, reward_predictor)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return rewards


if __name__ == "__main__":
    OBS_DIM = 8
    ACTION_DIM = 4
    LATENT_DIM = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load("data/checkpoint_final.pt", map_location=device)
    encoder = Encoder(OBS_DIM, LATENT_DIM).to(device)
    predictor = Predictor(LATENT_DIM, ACTION_DIM).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    encoder.eval()
    predictor.eval()

    reward_predictor = RewardPredictor(LATENT_DIM, ACTION_DIM).to(device)
    reward_predictor.load_state_dict(
        torch.load("data/reward_predictor_final.pt", map_location=device)
    )
    reward_predictor.eval()

    N_RUNS = 10
    env = gym.make("LunarLander-v3", render_mode="human")
    all_rewards = []
    for run in range(N_RUNS):
        obs, _ = env.reset()
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                action = cem(obs_tensor, encoder, predictor, reward_predictor, 100, 100)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        all_rewards.append(total_reward)
        print(f"run {run + 1:2d}/{N_RUNS} | total reward: {total_reward:.2f}")
    print(f"\nmean reward: {sum(all_rewards) / N_RUNS:.2f}")
