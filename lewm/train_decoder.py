import random
import argparse
import torch
import torch.nn.functional as F
import gymnasium as gym
import wandb
import numpy as np
from lewm.encoder import Encoder
from lewm.decoder import Decoder
from lewm.predictor import Predictor
from lewm.utils import collect_data, signed_log
from lewm.params import OBS_DIM, ACTION_DIM, LATENT_DIM


def make_batch(samples: list[tuple], device: torch.device) -> tuple:
    prev_obs = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32).to(device)
    obs = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32).to(device)
    prev_r = signed_log(torch.tensor([s[2] for s in samples], dtype=torch.float32)).to(device)
    actions = torch.tensor([s[3] for s in samples], dtype=torch.long).to(device)
    next_obs = torch.tensor(np.array([s[4] for s in samples]), dtype=torch.float32).to(device)
    next_r = signed_log(torch.tensor([s[5] for s in samples], dtype=torch.float32)).to(device)
    enc_in = torch.cat([prev_obs, obs, prev_r.unsqueeze(-1)], dim=-1)
    dec_target = torch.cat([next_obs, next_r.unsqueeze(-1)], dim=-1)
    return enc_in, actions, dec_target


def train_step(encoder, predictor, decoder, optimizer,
               enc_in: torch.Tensor, actions: torch.Tensor, dec_target: torch.Tensor) -> float:
    with torch.no_grad():
        z_next = predictor(encoder(enc_in), actions)
    out = decoder(z_next)
    loss = F.mse_loss(out, dec_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def train_epoch(encoder, predictor, decoder, optimizer, data, device, batch_size: int) -> float:
    random.shuffle(data)
    losses = []
    for i in range(0, len(data) - batch_size, batch_size):
        enc_in, actions, dec_target = make_batch(data[i:i + batch_size], device)
        losses.append(train_step(encoder, predictor, decoder, optimizer, enc_in, actions, dec_target))
    return sum(losses) / len(losses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notrack", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="data/checkpoint_final.pt")
    args = parser.parse_args()

    N_COLLECT = 100_000
    BATCH_SIZE = 256
    LR = 1e-4
    N_EPOCHS = 400
    SAVE_EVERY = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    wandb.init(
        project="lewm-lunarlander",
        config={"latent_dim": LATENT_DIM, "lr": LR, "n_collect": N_COLLECT},
        mode="disabled" if args.notrack else "online",
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    encoder = Encoder(OBS_DIM * 2 + 1, LATENT_DIM).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    predictor = Predictor(LATENT_DIM, ACTION_DIM).to(device)
    predictor.load_state_dict(ckpt["predictor"])
    predictor.eval()

    decoder = Decoder(LATENT_DIM, OBS_DIM).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)

    env = gym.make("LunarLander-v3")
    print(f"collecting {N_COLLECT} steps...")
    data = collect_data(env, N_COLLECT)
    print(f"collected {len(data)} transitions")

    for epoch in range(N_EPOCHS):
        loss = train_epoch(encoder, predictor, decoder, optimizer, data, device, BATCH_SIZE)
        print(f"\re{epoch:3d} | dec {loss:.4f}", end="")
        wandb.log({"decoder_loss": loss}, step=epoch)
        if not args.nosave and epoch % SAVE_EVERY == 0 and epoch > 0:
            torch.save(decoder.state_dict(), f"data/decoder_ckpt_{epoch}.pt")

    if not args.nosave:
        torch.save(decoder.state_dict(), "data/decoder_final.pt")
    print("\ndone.")
    wandb.finish()
