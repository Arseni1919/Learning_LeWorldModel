import random
import argparse
import torch
import torch.nn.functional as F
import gymnasium as gym
import wandb
import numpy as np
from lewm.encoder import Encoder
from lewm.predictor import Predictor
from lewm.sigreg import SIGReg
from lewm.utils import collect_data


def make_batch(samples: list[tuple], device: torch.device) -> tuple:
    prev_obs = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32).to(device)
    obs = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32).to(device)
    actions = torch.tensor([s[2] for s in samples], dtype=torch.long).to(device)
    next_obs = torch.tensor(np.array([s[3] for s in samples]), dtype=torch.float32).to(device)
    rewards = torch.tensor([s[4] for s in samples], dtype=torch.float32).to(device) / 100.0
    terminated = torch.tensor([s[5] for s in samples], dtype=torch.float32).to(device)
    return prev_obs, obs, actions, next_obs, rewards, terminated


def latent_stats(z: torch.Tensor) -> tuple:
    var = z.detach().var(dim=0)
    return var.mean().item(), (var < 0.01).sum().item()


def train_step(encoder, predictor, sigreg, optimizer, batch, lam: float) -> tuple:
    prev_obs, obs, actions, next_obs, _, _ = batch
    z = encoder(obs)
    z_next = encoder(next_obs)
    z_hat_next = predictor(z, actions)
    pred_loss = F.mse_loss(z_hat_next, z_next)
    reg_loss = sigreg(torch.stack([z, z_next]))
    loss = pred_loss + lam * reg_loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), pred_loss.item(), reg_loss.item(), *latent_stats(z)


def train_epoch(encoder, predictor, sigreg, optimizer, data, device,
                batch_size: int, lam: float) -> dict:
    random.shuffle(data)
    losses, pred_losses, reg_losses, vars_, deads = [], [], [], [], []
    for i in range(0, len(data) - batch_size, batch_size):
        batch = make_batch(data[i:i + batch_size], device)
        loss, pred_loss, reg_loss, mean_var, dead_dims = train_step(
            encoder, predictor, sigreg, optimizer, batch, lam)
        losses.append(loss)
        pred_losses.append(pred_loss)
        reg_losses.append(reg_loss)
        vars_.append(mean_var)
        deads.append(dead_dims)
    return {
        "loss": sum(losses) / len(losses),
        "pred_loss": sum(pred_losses) / len(pred_losses),
        "reg_loss": sum(reg_losses) / len(reg_losses),
        "mean_var": sum(vars_) / len(vars_),
        "dead_dims": sum(deads) / len(deads),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notrack", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    args = parser.parse_args()

    OBS_DIM = 8
    ACTION_DIM = 4
    LATENT_DIM = 16
    N_COLLECT = 10_000
    BATCH_SIZE = 256
    LR = 3e-4
    LAMBDA = 1.0
    N_EPOCHS = 100
    SAVE_EVERY = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    wandb.init(
        project="lewm-lunarlander",
        config={"latent_dim": LATENT_DIM, "lr": LR, "lambda": LAMBDA,
                "batch_size": BATCH_SIZE, "n_collect": N_COLLECT},
        mode="disabled" if args.notrack else "online",
    )

    env = gym.make("LunarLander-v3")
    encoder = Encoder(OBS_DIM, LATENT_DIM).to(device)
    predictor = Predictor(LATENT_DIM, ACTION_DIM).to(device)
    sigreg = SIGReg().to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=LR
    )

    print(f"collecting {N_COLLECT} steps...")
    data = collect_data(env, N_COLLECT)
    print(f"collected {len(data)} transitions")

    step = 0
    for epoch in range(N_EPOCHS):
        random.shuffle(data)
        for i in range(0, len(data) - BATCH_SIZE, BATCH_SIZE):
            batch = make_batch(data[i:i + BATCH_SIZE], device)
            loss, pred_loss, reg_loss, mean_var, dead_dims = train_step(
                encoder, predictor, sigreg, optimizer, batch, LAMBDA)
            step += 1
            msg = (f"e{epoch:3d} s{step:5d} | loss {loss:.4f}"
                   f" | pred {pred_loss:.4f} | sig {reg_loss:.4f}"
                   f" | var {mean_var:.3f} | dead {dead_dims}")
            print(f'\r{msg}', end="")
            wandb.log({
                "loss": loss, "pred_loss": pred_loss, "sigreg_loss": reg_loss,
                "latent_mean_var": mean_var, "latent_dead_dims": dead_dims,
            }, step=step)
            if not args.nosave and step % SAVE_EVERY == 0 and step > 1000:
                ckpt = {"encoder": encoder.state_dict(), "predictor": predictor.state_dict()}
                torch.save(ckpt, f"data/checkpoint_{step}.pt")

    if not args.nosave:
        torch.save({
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
        }, "data/checkpoint_final.pt")
    wandb.finish()
