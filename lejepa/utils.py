import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def evaluate_nd(encoder, makers, labels, device, colors=None):
    encoder.eval()
    n = len(labels)
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(n)]
    all_z = []
    with torch.no_grad():
        for maker in makers:
            imgs = torch.stack([maker() for _ in range(20)]).to(device)
            all_z.append(encoder(imgs).cpu())
    Z = torch.cat(all_z)
    _, _, V = torch.pca_lowrank(Z, q=2)
    Z2 = Z @ V
    fig, ax = plt.subplots(figsize=(5, 5), num="latent", clear=True)
    for i, (label, color) in enumerate(zip(labels, colors)):
        clr = color.tolist() if hasattr(color, 'tolist') else color
        ax.scatter(Z2[i*20:(i+1)*20, 0], Z2[i*20:(i+1)*20, 1], color=clr, label=label, alpha=0.7)
    ax.legend()
    ax.set_title("PCA projection")
    plt.tight_layout()
    plt.pause(0.1)


def evaluate_3d(encoder, makers, labels, device, colors=None):
    encoder.eval()
    n = len(labels)
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(n)]
    fig = plt.figure(num="latent", clear=True)
    ax = fig.add_subplot(111, projection='3d')
    with torch.no_grad():
        for maker, label, color in zip(makers, labels, colors):
            imgs = torch.stack([maker() for _ in range(20)]).to(device)
            z = encoder(imgs)
            clr = color.tolist() if hasattr(color, 'tolist') else color
            ax.scatter(z[:, 0].cpu(), z[:, 1].cpu(), z[:, 2].cpu(), color=clr, label=label, alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.pause(0.1)


def evaluate_2d(encoder, makers, labels, device, colors=None):
    encoder.eval()
    n = len(labels)
    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(n)]
    fig, ax = plt.subplots(figsize=(5, 5), num="latent", clear=True)
    with torch.no_grad():
        for maker, label, color in zip(makers, labels, colors):
            imgs = torch.stack([maker() for _ in range(20)]).to(device)
            z = encoder(imgs)
            clr = color.tolist() if hasattr(color, 'tolist') else color
            ax.scatter(z[:, 0].cpu(), z[:, 1].cpu(), color=clr, label=label, alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.pause(0.1)


class Normalize(nn.Module):
    def forward(self, x):
        return x * 2 - 1


class EncoderChannelMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            Normalize(),
            nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.mean(dim=1)
        return x.flatten(1)
