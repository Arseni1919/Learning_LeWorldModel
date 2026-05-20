import random
import argparse
import torch
import torch.nn.functional as F
import gymnasium as gym
import wandb
import numpy as np
from lewm.encoder import Encoder
from lewm.predictor import Predictor
from lewm.reward_predictor import RewardPredictor
from lewm.train import collect_data


def make_batch(samples: list[tuple], device: torch.device) -> tuple:
    prev_obs = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32).to(device)
    obs = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32).to(device)
    actions = torch.tensor([s[2] for s in samples], dtype=torch.long).to(device)
    rewards = torch.tensor([s[4] for s in samples], dtype=torch.float32).to(device) / 100.0
    terminated = torch.tensor([s[5] for s in samples], dtype=torch.float32).to(device)
    return prev_obs, obs, actions, rewards, terminated


def train_step(encoder, reward_predictor, optimizer, batch) -> float:
    prev_obs, obs, actions, rewards, terminated = batch
    with torch.no_grad():
        z_prev = encoder(prev_obs)
        z = encoder(obs)
    r_hat = reward_predictor(z_prev, z, actions, terminated)
    loss = F.mse_loss(r_hat, rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notrack", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="data/checkpoint_final.pt")
    args = parser.parse_args()

    OBS_DIM = 8
    ACTION_DIM = 4
    LATENT_DIM = 16
    N_COLLECT = 10_000
    BATCH_SIZE = 256
    LR = 3e-4
    N_EPOCHS = 100
    SAVE_EVERY = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    wandb.init(
        project="lewm-lunarlander",
        config={"latent_dim": LATENT_DIM, "lr": LR, "n_collect": N_COLLECT},
        mode="disabled" if args.notrack else "online",
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    encoder = Encoder(OBS_DIM, LATENT_DIM).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    reward_predictor = RewardPredictor(LATENT_DIM, ACTION_DIM).to(device)
    optimizer = torch.optim.Adam(reward_predictor.parameters(), lr=LR)

    env = gym.make("LunarLander-v3")
    print(f"collecting {N_COLLECT} steps...")
    data = collect_data(env, N_COLLECT)
    print(f"collected {len(data)} transitions")

    step = 0
    for epoch in range(N_EPOCHS):
        random.shuffle(data)
        for i in range(0, len(data) - BATCH_SIZE, BATCH_SIZE):
            batch = make_batch(data[i:i + BATCH_SIZE], device)
            loss = train_step(encoder, reward_predictor, optimizer, batch)
            step += 1
            print(f"\re{epoch:3d} s{step:5d} | rew {loss:.4f}", end="")
            wandb.log({"reward_loss": loss}, step=step)
            if not args.nosave and step % SAVE_EVERY == 0:
                torch.save(reward_predictor.state_dict(), f"data/reward_ckpt_{step}.pt")

    if not args.nosave:
        torch.save(reward_predictor.state_dict(), "data/reward_predictor_final.pt")
    print("\ndone.")
    wandb.finish()
