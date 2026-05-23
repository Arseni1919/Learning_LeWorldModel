import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from lewm.sigreg import SIGReg
from lejepa.encoder import Encoder
from lejepa.fig_gen import make_square_image, make_circle_image, make_triangle_image

MAKERS = [make_square_image, make_circle_image, make_triangle_image]
LABELS = ["Square", "Circle", "Triangle"]
BATCH_SIZE = 64
LR = 1e-4
ETA_MIN = 1e-5
LAMBDA = 0.1
N_EPOCHS = 500
SAVE_EVERY = 10
DATA_SIZE = 7680


def plot_pair(img1, img2):
    i1 = img1[0] if img1.dim() == 4 else img1
    i2 = img2[0] if img2.dim() == 4 else img2
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(i1.cpu().permute(1, 2, 0).numpy())
    axes[0].axis("off")
    axes[1].imshow(i2.cpu().permute(1, 2, 0).numpy())
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


def make_dataset(n=1000):
    data = []
    for label, maker in enumerate(MAKERS):
        for _ in range(n):
            data.append((maker(), label))
    return data


def make_batches(data, batch_size, device):
    by_class = {}
    for img, label in data:
        by_class.setdefault(label, []).append(img)
    batches = []
    for imgs in by_class.values():
        idx = torch.randperm(len(imgs))
        imgs = [imgs[i] for i in idx]
        for i in range(0, len(imgs) - batch_size, batch_size):
            batches.append(imgs[i:i + batch_size])
    idx = torch.randperm(len(batches))
    batches = [batches[i] for i in idx]
    for batch in batches:
        yield torch.stack(batch).to(device)


def latent_stats(z):
    var = z.detach().var(dim=0)
    return var.mean().item(), (var < 0.01).sum().item()


def evaluate(encoder, axes, device):
    encoder.eval()
    with torch.no_grad():
        for ax, maker, label in zip(axes, MAKERS, LABELS):
            img = maker().unsqueeze(0).to(device)
            out = encoder(img).squeeze().cpu().numpy()
            ax.cla()
            ax.imshow(out.reshape(8, 8), cmap="gray")
            ax.set_title(label)
            ax.axis("off")
    plt.pause(0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notrack", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    wandb.init(
        project="lejepa-shapes",
        config={"lr": LR, "lambda": LAMBDA, "batch_size": BATCH_SIZE, "n_epochs": N_EPOCHS},
        mode="disabled" if args.notrack else "online",
    )

    data = make_dataset(DATA_SIZE)
    encoder = Encoder().to(device)
    sigreg = SIGReg().to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=ETA_MIN)

    plt.ion()
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    fig.suptitle("Encoder outputs")

    step = 0
    for epoch in range(N_EPOCHS):
        encoder.train()
        for batch_idx, imgs in enumerate(make_batches(data, BATCH_SIZE, device)):
            Z = encoder(imgs).flatten(1)
            z_mean = Z.mean(dim=0, keepdim=True)
            pred_loss = F.mse_loss(Z, z_mean.expand_as(Z))
            reg_loss = sigreg(Z.unsqueeze(0))
            loss = pred_loss + LAMBDA * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_var, dead_dims = latent_stats(Z)
            lr = optimizer.param_groups[0]["lr"]
            print(f"\re{epoch:3d} b{batch_idx:3d} | loss {loss:.4f}"
                  f" | pred {pred_loss:.4f} | sig {reg_loss:.4f}"
                  f" | var {mean_var:.4f} | dead {dead_dims} | lr {lr:.2e}", end="")
            wandb.log({
                "loss": loss.item(), "pred_loss": pred_loss.item(),
                "reg_loss": reg_loss.item(), "mean_var": mean_var, "dead_dims": dead_dims,
            }, step=step)
            step += 1
            # if (batch_idx + 1) % 10 == 0:
            #     evaluate(encoder, axes, device)
            #     encoder.train()
        scheduler.step()
        wandb.log({"lr": scheduler.get_last_lr()[0]}, step=step)
        if not args.nosave and epoch % SAVE_EVERY == 0 and epoch > 0:
            torch.save({"encoder": encoder.state_dict()}, f"data/lejepa_checkpoint.pt")
    print()
    if not args.nosave:
        torch.save({"encoder": encoder.state_dict()}, "data/lejepa_checkpoint_final.pt")
    plt.ioff()
    plt.show()
    wandb.finish()
