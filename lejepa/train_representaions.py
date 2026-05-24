import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
from lewm.sigreg import SIGReg
from lejepa.encoder import Encoder
from lejepa.predictor import PredictorConcat as Predictor
from lejepa.fig_gen import make_square_image, make_circle_image, make_triangle_image, RAINBOW_COLORS, COLOR_NAMES
from lejepa.utils import evaluate_2d, evaluate_3d, evaluate_nd

MAKERS = [make_square_image, make_circle_image, make_triangle_image]
LABELS = ["Square", "Circle", "Triangle"]
BATCH_SIZE = 64
LR = 1e-3
ETA_MIN = 1e-5
LAMBDA = 0.1
N_EPOCHS = 100
LATENT_DIM = 16
SAVE_EVERY = 10
DATA_SIZE = BATCH_SIZE * 40
DATA_SOURCE = "shapes"  # "shapes" or "colors"
# DATA_SOURCE = "colors"  # "shapes" or "colors"


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


def make_color_dataset(n=1000):
    data = []
    for label, color in enumerate(RAINBOW_COLORS):
        for _ in range(n):
            maker = MAKERS[torch.randint(0, len(MAKERS), (1,)).item()]
            data.append((maker(fg=color), label))
    return data


def make_batches(data, batch_size, device):
    by_class = {}
    for img, label in data:
        by_class.setdefault(label, []).append(img)
    n_classes = len(by_class)
    for _ in range(len(data) // batch_size):
        labels = torch.randint(0, n_classes, (batch_size,))
        one_hots = F.one_hot(labels, n_classes).float().to(device)
        imgs1, imgs2 = [], []
        for lbl in labels.tolist():
            cls_imgs = by_class[lbl]
            i, j = torch.randint(0, len(cls_imgs), (2,)).tolist()
            imgs1.append(cls_imgs[i])
            imgs2.append(cls_imgs[j])
        yield torch.stack(imgs1).to(device), torch.stack(imgs2).to(device), one_hots


def latent_stats(z):
    var = z.detach().var(dim=0)
    return var.mean().item(), (var < 0.01).sum().item()



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

    data = make_color_dataset(DATA_SIZE) if DATA_SOURCE == "colors" else make_dataset(DATA_SIZE)
    n_classes = len(set(label for _, label in data))
    encoder = Encoder(latent_dim=LATENT_DIM).to(device)
    predictor = Predictor(latent_dim=LATENT_DIM, n_classes=n_classes).to(device)
    sigreg = SIGReg().to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=ETA_MIN)

    plt.ion()

    step = 0
    for epoch in range(N_EPOCHS):
        encoder.train()
        for batch_idx, (imgs1, imgs2, one_hots) in enumerate(make_batches(data, BATCH_SIZE, device)):
            z1 = encoder(imgs1)
            z2 = encoder(imgs2)
            z_pred = predictor(z1, one_hots)
            pred_loss = F.mse_loss(z_pred, z2)
            reg_loss = sigreg(torch.stack([z_pred, z2]))
            loss = pred_loss + LAMBDA * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mean_var, dead_dims = latent_stats(z2)
            lr = optimizer.param_groups[0]["lr"]
            print(f"\re{epoch:3d} b{batch_idx:3d} | loss {loss:.4f}"
                  f" | pred {pred_loss:.4f} | sig {reg_loss:.4f}"
                  f" | var {mean_var:.4f} | dead {dead_dims} | lr {lr:.2e}", end="")
            wandb.log({
                "loss": loss.item(), "pred_loss": pred_loss.item(),
                "reg_loss": reg_loss.item(), "mean_var": mean_var, "dead_dims": dead_dims,
            }, step=step)
            step += 1
        scheduler.step()
        if DATA_SOURCE == "colors":
            eval_makers = [lambda c=color: make_square_image(fg=c) for color in RAINBOW_COLORS]
            eval_labels, eval_colors = COLOR_NAMES, RAINBOW_COLORS
        else:
            eval_makers, eval_labels, eval_colors = MAKERS, LABELS, None
        if LATENT_DIM == 2:
            evaluate_2d(encoder, eval_makers, eval_labels, device, colors=eval_colors)
        elif LATENT_DIM == 3:
            evaluate_3d(encoder, eval_makers, eval_labels, device, colors=eval_colors)
        else:
            evaluate_nd(encoder, eval_makers, eval_labels, device, colors=eval_colors)
        encoder.train()
        wandb.log({"lr": scheduler.get_last_lr()[0]}, step=step)
        if not args.nosave and epoch % SAVE_EVERY == 0 and epoch > 0:
            torch.save({"encoder": encoder.state_dict()}, f"data/lejepa_checkpoint.pt")
    print()
    if not args.nosave:
        torch.save({"encoder": encoder.state_dict()}, "data/lejepa_checkpoint_final.pt")
    plt.ioff()
    plt.show()
    wandb.finish()
