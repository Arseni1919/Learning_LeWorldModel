import random
import argparse
import torch
import torch.nn.functional as F
import gymnasium as gym
import wandb
import numpy as np
from lewm.encoder import Encoder
from lewm.decoder import Decoder
from lewm.train_jepa import collect_data


def make_batch(samples: list[tuple], device: torch.device) -> torch.Tensor:
    return torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32).to(device)


def train_step(encoder, decoder, optimizer, obs) -> float:
    with torch.no_grad():
        z = encoder(obs)
    obs_hat = decoder(z)
    loss = F.mse_loss(obs_hat, obs)
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

    decoder = Decoder(LATENT_DIM, OBS_DIM).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)

    env = gym.make("LunarLander-v3")
    print(f"collecting {N_COLLECT} steps...")
    data = collect_data(env, N_COLLECT)
    print(f"collected {len(data)} transitions")

    step = 0
    for epoch in range(N_EPOCHS):
        random.shuffle(data)
        for i in range(0, len(data) - BATCH_SIZE, BATCH_SIZE):
            batch = make_batch(data[i:i + BATCH_SIZE], device)
            loss = train_step(encoder, decoder, optimizer, batch)
            step += 1
            print(f"\re{epoch:3d} s{step:5d} | dec {loss:.4f}", end="")
            wandb.log({"decoder_loss": loss}, step=step)
            if not args.nosave and step % SAVE_EVERY == 0:
                torch.save(decoder.state_dict(), f"data/decoder_ckpt_{step}.pt")

    if not args.nosave:
        torch.save(decoder.state_dict(), "data/decoder_final.pt")
    print("\ndone.")
    wandb.finish()
