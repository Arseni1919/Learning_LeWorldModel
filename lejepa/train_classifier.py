import argparse
import torch
import torch.nn.functional as F
import wandb
from lejepa.encoder import Encoder
from lejepa.fig_gen import make_square_image, make_circle_image, make_triangle_image

MAKERS = [make_square_image, make_circle_image, make_triangle_image]
BATCH_SIZE = 64
LR = 1e-3
ETA_MIN = 1e-5
N_EPOCHS = 50
DATA_SIZE = 10000
VAL_SPLIT = 0.2


def make_dataset(n):
    data = []
    for label, maker in enumerate(MAKERS):
        for _ in range(n):
            data.append((maker(), label))
    return data


def make_batches(data, batch_size, device):
    idx = torch.randperm(len(data)).tolist()
    for i in range(0, len(data) - batch_size, batch_size):
        batch = [data[idx[j]] for j in range(i, i + batch_size)]
        imgs = torch.stack([img for img, _ in batch]).to(device)
        labels = torch.tensor([label for _, label in batch], device=device)
        yield imgs, labels


def accuracy(encoder, data, device):
    encoder.eval()
    with torch.no_grad():
        imgs = torch.stack([img for img, _ in data]).to(device)
        labels = torch.tensor([label for _, label in data], device=device)
        return (encoder(imgs).argmax(dim=1) == labels).float().mean().item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notrack", action="store_true")
    parser.add_argument("--nosave", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    wandb.init(
        project="lejepa-classifier",
        config={"lr": LR, "batch_size": BATCH_SIZE, "n_epochs": N_EPOCHS, "data_size": DATA_SIZE},
        mode="disabled" if args.notrack else "online",
    )

    data = make_dataset(DATA_SIZE)
    n_val = int(len(data) * VAL_SPLIT)
    idx = torch.randperm(len(data)).tolist()
    val_data = [data[i] for i in idx[:n_val]]
    train_data = [data[i] for i in idx[n_val:]]

    encoder = Encoder(latent_dim=3).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=ETA_MIN)

    best_val_acc = 0.0
    for epoch in range(N_EPOCHS):
        encoder.train()
        batches = list(make_batches(train_data, BATCH_SIZE, device))
        for batch_idx, (imgs, labels) in enumerate(batches):
            loss = F.cross_entropy(encoder(imgs), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"\re{epoch:3d} b{batch_idx + 1:3d}/{len(batches)} | loss {loss:.4f}", end="")
        scheduler.step()
        train_acc = accuracy(encoder, train_data, device)
        val_acc = accuracy(encoder, val_data, device)
        if not args.nosave and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"encoder": encoder.state_dict()}, "data/lejepa_classifier_final.pt")
        print(f"\re{epoch:3d} | loss {loss:.4f} | train_acc {train_acc:.3f} | val_acc {val_acc:.3f}", end="")
        wandb.log({"loss": loss.item(), "train_acc": train_acc, "val_acc": val_acc}, step=epoch)
        encoder.train()
    print()
    wandb.finish()
