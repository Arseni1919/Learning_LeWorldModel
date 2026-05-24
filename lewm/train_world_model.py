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
from lewm.utils import collect_data, signed_log
from lewm.params import OBS_DIM, ACTION_DIM, LATENT_DIM


def make_batch(samples: list[tuple], device: torch.device) -> tuple:
    prev_obs = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32).to(device)
    obs = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32).to(device)
    prev_r = signed_log(torch.tensor([s[2] for s in samples], dtype=torch.float32)).to(device)
    actions = torch.tensor([s[3] for s in samples], dtype=torch.long).to(device)
    next_obs = torch.tensor(np.array([s[4] for s in samples]), dtype=torch.float32).to(device)
    next_r = signed_log(torch.tensor([s[5] for s in samples], dtype=torch.float32)).to(device)
    return prev_obs, obs, prev_r, actions, next_obs, next_r


def latent_stats(z: torch.Tensor) -> tuple:
    var = z.detach().var(dim=0)
    return var.mean().item(), (var < 0.01).sum().item()


def train_step(encoder, predictor, sigreg, optimizer, batch, lam: float) -> tuple:
    prev_obs, obs, prev_r, actions, next_obs, next_r = batch
    enc_in = torch.cat([prev_obs, obs, prev_r.unsqueeze(-1)], dim=-1)
    enc_next_in = torch.cat([obs, next_obs, next_r.unsqueeze(-1)], dim=-1)
    z = encoder(enc_in)
    z_next = encoder(enc_next_in)
    z_hat_next = predictor(z, actions)
    pred_loss = F.mse_loss(z_hat_next, z_next)
    reg_loss = sigreg(torch.stack([z, z_next]))
    loss = pred_loss + lam * reg_loss
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(predictor.parameters()), max_norm=1.0
    )
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

    N_COLLECT = 1_000
    BATCH_SIZE = 256
    LR = 1e-4
    LAMBDA = 1.0
    N_EPOCHS = 1200
    SAVE_EVERY = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    wandb.init(
        project="lewm-lunarlander",
        config={"latent_dim": LATENT_DIM, "lr": LR, "lambda": LAMBDA,
                "batch_size": BATCH_SIZE, "n_collect": N_COLLECT},
        mode="disabled" if args.notrack else "online",
    )

    env = gym.make("LunarLander-v3")
    encoder = Encoder(OBS_DIM * 2 + 1, LATENT_DIM).to(device)
    predictor = Predictor(LATENT_DIM, ACTION_DIM).to(device)
    sigreg = SIGReg().to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()), lr=LR
    )

    print(f"collecting {N_COLLECT} steps...")
    data = collect_data(env, N_COLLECT)
    print(f"collected {len(data)} transitions")

    for epoch in range(N_EPOCHS):
        stats = train_epoch(encoder, predictor, sigreg, optimizer, data, device, BATCH_SIZE, LAMBDA)
        msg = (f"e{epoch:3d} | loss {stats['loss']:.4f}"
               f" | pred {stats['pred_loss']:.4f} | sig {stats['reg_loss']:.4f}"
               f" | var {stats['mean_var']:.3f} | dead {stats['dead_dims']:.0f}")
        print(f'\r{msg}', end="")
        wandb.log({
            "loss": stats["loss"], "pred_loss": stats["pred_loss"],
            "sigreg_loss": stats["reg_loss"], "latent_mean_var": stats["mean_var"],
            "latent_dead_dims": stats["dead_dims"],
        }, step=epoch)
        if not args.nosave and epoch % SAVE_EVERY == 0 and epoch > 0:
            ckpt = {"encoder": encoder.state_dict(), "predictor": predictor.state_dict()}
            torch.save(ckpt, f"data/checkpoint_{epoch}.pt")

    if not args.nosave:
        torch.save({
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
        }, "data/checkpoint_final.pt")
    wandb.finish()
