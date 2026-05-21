import time
import heapq
import argparse
import torch
import gymnasium as gym
from tqdm import tqdm
from lewm.encoder import Encoder
from lewm.decoder import Decoder
from lewm.predictor import Predictor
from lewm.utils import signed_log


def _enc_input(prev_obs: torch.Tensor, obs: torch.Tensor,
               prev_reward_log: float, device) -> torch.Tensor:
    r = torch.tensor([prev_reward_log], dtype=torch.float32, device=device).expand(obs.shape[0], 1)
    return torch.cat([prev_obs, obs, r], dim=-1)


def cem(obs: torch.Tensor, encoder, predictor, decoder,
        prev_obs: torch.Tensor = None,
        prev_reward_log: float = 0.0,
        H: int = 10, N: int = 100, n_iter: int = 5,
        elite_frac: float = 0.1, smoothing: float = 0.1) -> int:
    device = obs.device
    ACTION_DIM = 4
    if prev_obs is None:
        prev_obs = torch.zeros_like(obs)
    enc_in = _enc_input(prev_obs.unsqueeze(0), obs.unsqueeze(0), prev_reward_log, device)
    z0 = encoder(enc_in).expand(N, -1)
    n_elite = max(1, int(N * elite_frac))
    probs = torch.ones(H, ACTION_DIM, device=device) / ACTION_DIM
    actions = None
    for _ in range(n_iter):
        actions = torch.stack([
            torch.multinomial(probs[h], N, replacement=True) for h in range(H)
        ]).T  # (N, H)
        z = z0.clone()
        scores = torch.zeros(N, device=device)
        for h in range(H):
            z_next = predictor(z, actions[:, h])
            scores += decoder(z_next)[:, -1]
            z = z_next
        elite = actions[scores.topk(n_elite).indices]  # (n_elite, H)
        counts = torch.zeros(H, ACTION_DIM, device=device)
        for h in range(H):
            counts[h].scatter_add_(0, elite[:, h], torch.ones(n_elite, device=device))
        probs = (1 - smoothing) * counts / n_elite + smoothing / ACTION_DIM
    return actions[scores.argmax(), 0].item()


def a_star(obs: torch.Tensor, encoder, predictor, decoder,
           prev_obs: torch.Tensor = None,
           prev_reward_log: float = 0.0,
           action_dim: int = 4, max_nodes: int = 200) -> int:
    device = obs.device
    if prev_obs is None:
        prev_obs = torch.zeros_like(obs)
    enc_in = _enc_input(prev_obs.unsqueeze(0), obs.unsqueeze(0), prev_reward_log, device)
    z0 = encoder(enc_in).squeeze(0)
    heap = []
    counter = 0
    for a in range(action_dim):
        action_t = torch.tensor([a], device=device)
        z_next = predictor(z0.unsqueeze(0), action_t).squeeze(0)
        r = decoder(z_next.unsqueeze(0))[0, -1].item()
        heapq.heappush(heap, (-r, counter, z_next, a))
        counter += 1
    best_g, best_first = float("-inf"), 0
    for _ in range(max_nodes):
        if not heap:
            break
        neg_g, _, z, first_action = heapq.heappop(heap)
        g = -neg_g
        if g > best_g:
            best_g, best_first = g, first_action
        for a in range(action_dim):
            action_t = torch.tensor([a], device=device)
            z_next = predictor(z.unsqueeze(0), action_t).squeeze(0)
            r = decoder(z_next.unsqueeze(0))[0, -1].item()
            heapq.heappush(heap, (-(g + r), counter, z_next, first_action))
            counter += 1
    return best_first


def demo(encoder, predictor, decoder, device):
    env = gym.make("LunarLander-v3", render_mode="human")
    obs, _ = env.reset()
    terminated = truncated = False
    total_reward = 0.0
    prev_obs = obs.copy()
    prev_reward_log = 0.0
    while not (terminated or truncated):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
        prev_obs_tensor = torch.tensor(prev_obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            action = cem(obs_tensor, encoder, predictor, decoder,
                         prev_obs=prev_obs_tensor, prev_reward_log=prev_reward_log)
        prev_obs = obs.copy()
        obs, reward, terminated, truncated, _ = env.step(action)
        prev_reward_log = signed_log(reward)
        total_reward += reward
    env.close()
    time.sleep(1)
    print(f"  demo | total reward: {total_reward:.2f}")


def evaluate(env, encoder, predictor, decoder, n_episodes: int, device) -> list[float]:
    rewards = []
    for _ in tqdm(range(n_episodes), desc="evaluating", unit="episode"):
        obs, _ = env.reset()
        total_reward = 0.0
        terminated = truncated = False
        prev_obs = obs.copy()
        prev_reward_log = 0.0
        while not (terminated or truncated):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            prev_obs_tensor = torch.tensor(prev_obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                action = cem(obs_tensor, encoder, predictor, decoder,
                             prev_obs=prev_obs_tensor, prev_reward_log=prev_reward_log)
            prev_obs = obs.copy()
            obs, reward, terminated, truncated, _ = env.step(action)
            prev_reward_log = signed_log(reward)
            total_reward += reward
        rewards.append(total_reward)
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=["cem", "a_star"], default="a_star")
    args = parser.parse_args()

    OBS_DIM = 8
    ACTION_DIM = 4
    LATENT_DIM = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load("data/checkpoint_final.pt", map_location=device)
    encoder = Encoder(OBS_DIM * 2 + 1, LATENT_DIM).to(device)
    predictor = Predictor(LATENT_DIM, ACTION_DIM).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    encoder.eval()
    predictor.eval()

    decoder = Decoder(LATENT_DIM, OBS_DIM).to(device)
    decoder.load_state_dict(torch.load("data/decoder_final.pt", map_location=device))
    decoder.eval()

    planner = cem if args.planner == "cem" else a_star
    print(f"planner: {args.planner}")

    N_RUNS = 10
    env = gym.make("LunarLander-v3", render_mode="human")
    all_rewards = []
    for run in range(N_RUNS):
        obs, _ = env.reset()
        total_reward = 0.0
        terminated = truncated = False
        prev_obs = obs.copy()
        prev_reward_log = 0.0
        while not (terminated or truncated):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            prev_obs_tensor = torch.tensor(prev_obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                action = planner(obs_tensor, encoder, predictor, decoder,
                                 prev_obs=prev_obs_tensor, prev_reward_log=prev_reward_log)
            prev_obs = obs.copy()
            obs, reward, terminated, truncated, _ = env.step(action)
            prev_reward_log = signed_log(reward)
            total_reward += reward
        all_rewards.append(total_reward)
        print(f"run {run + 1:2d}/{N_RUNS} | total reward: {total_reward:.2f}")
    print(f"\nmean reward: {sum(all_rewards) / N_RUNS:.2f}")
