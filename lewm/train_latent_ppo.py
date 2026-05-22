import argparse
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from lewm.encoder import Encoder
from lewm.decoder import Decoder
from lewm.predictor import Predictor
from lewm.utils import signed_log, signed_exp
from lewm.params import OBS_DIM, ACTION_DIM, LATENT_DIM


def make_enc_in(prev_obs: np.ndarray, obs: np.ndarray, prev_reward_log: float) -> torch.Tensor:
    return torch.cat([
        torch.tensor(prev_obs, dtype=torch.float32),
        torch.tensor(obs, dtype=torch.float32),
        torch.tensor([prev_reward_log], dtype=torch.float32),
    ]).unsqueeze(0)


def load_world_model(checkpoint: str, decoder_path: str, device: torch.device):
    encoder = Encoder(OBS_DIM * 2 + 1, LATENT_DIM).to(device)
    predictor = Predictor(LATENT_DIM, ACTION_DIM).to(device)
    decoder = Decoder(LATENT_DIM, OBS_DIM).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    encoder.eval()
    predictor.eval()
    decoder.eval()
    return encoder, predictor, decoder


class LatentEnv(gym.Env):
    def __init__(self, encoder, predictor, decoder, real_env, max_steps: int = 500):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.decoder = decoder
        self.real_env = real_env
        self.max_steps = max_steps
        self.device = next(encoder.parameters()).device
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(LATENT_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(ACTION_DIM)
        self.z = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        obs, _ = self.real_env.reset(seed=seed)
        enc_in = make_enc_in(obs, obs, 0.0).to(self.device)
        with torch.no_grad():
            self.z = self.encoder(enc_in).squeeze(0)
        self.step_count = 0
        return self.z.cpu().numpy(), {}

    def step(self, action):
        action_t = torch.tensor([action], dtype=torch.long, device=self.device)
        with torch.no_grad():
            z_next = self.predictor(self.z.unsqueeze(0), action_t).squeeze(0)
            z_next = torch.nan_to_num(z_next, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10, 10)
            dec_out = self.decoder(z_next.unsqueeze(0)).squeeze(0)
        reward = float(signed_exp(dec_out[-1].item()))
        if not torch.isfinite(torch.tensor(reward)):
            reward = 0.0
        self.z = z_next
        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        return self.z.cpu().numpy(), reward, False, truncated, {}


def evaluate_in_real_env(policy, encoder, device, n_episodes: int = 10,
                         render: bool = True) -> list[float]:
    env = gym.make("LunarLander-v3", render_mode="human" if render else None)
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        prev_obs = obs.copy()
        prev_reward_log = 0.0
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            enc_in = make_enc_in(prev_obs, obs, prev_reward_log).to(device)
            with torch.no_grad():
                z = encoder(enc_in).squeeze(0)
            action, _ = policy.predict(z.cpu().numpy(), deterministic=True)
            prev_obs = obs.copy()
            obs, reward, terminated, truncated, _ = env.step(int(action))
            prev_reward_log = signed_log(reward)
            total_reward += reward
        rewards.append(total_reward)
    env.close()
    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="data/checkpoint_final.pt")
    parser.add_argument("--decoder", type=str, default="data/decoder_final.pt")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--nosave", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    encoder, predictor, decoder = load_world_model(args.checkpoint, args.decoder, device)
    real_env = gym.make("LunarLander-v3")
    latent_env = LatentEnv(encoder, predictor, decoder, real_env, max_steps=args.max_steps)

    policy = PPO("MlpPolicy", latent_env, verbose=1, device=str(device))
    policy.learn(total_timesteps=args.total_timesteps)

    if not args.nosave:
        policy.save("data/ppo_latent")

    print("\nevaluating in real env...")
    rewards = evaluate_in_real_env(policy, encoder, device, n_episodes=args.eval_episodes)
    mean_r = sum(rewards) / len(rewards)
    n_success = sum(r >= 200 for r in rewards)
    print(f"mean reward:  {mean_r:.2f}")
    print(f"success rate: {n_success}/{args.eval_episodes}")
    print(f"rewards:      {[f'{r:.0f}' for r in rewards]}")
