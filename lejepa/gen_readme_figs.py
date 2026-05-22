import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from lejepa.encoder import Encoder
from lejepa.fig_gen import make_square_image, make_circle_image, make_triangle_image

MAKERS = [make_square_image, make_circle_image, make_triangle_image]
LABELS = ["Square", "Circle", "Triangle"]
COLORS = ["steelblue", "tomato", "seagreen"]


def save_shapes_grid():
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    for row, (maker, label) in enumerate(zip(MAKERS, LABELS)):
        for col in range(5):
            ax = axes[row, col]
            ax.imshow(maker().permute(1, 2, 0).numpy())
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(label, fontsize=12, rotation=90, labelpad=10)
    plt.tight_layout()
    plt.savefig("pics/lejepa_shapes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("saved pics/lejepa_shapes.png")


def save_projections():
    N_SAMPLES = 500
    M = 64

    device = torch.device("cpu")
    encoder = Encoder()
    ckpt = torch.load("data/lejepa_checkpoint_final.pt", map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    encoder.eval()

    imgs_by_class = [[maker() for _ in range(N_SAMPLES)] for maker in MAKERS]
    with torch.no_grad():
        Z_by_class = [encoder(torch.stack(imgs)).flatten(1) for imgs in imgs_by_class]

    latent_dim = Z_by_class[0].shape[1]
    torch.manual_seed(0)
    U = torch.nn.functional.normalize(torch.randn(latent_dim, M), dim=0)

    fig, axes = plt.subplots(8, 8, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        for Z_cls, label, color in zip(Z_by_class, LABELS, COLORS):
            H_cls = (Z_cls @ U[:, i]).numpy()
            ax.hist(H_cls, bins=40, density=True, alpha=0.5, color=color, label=label)
        x = np.linspace(-3, 3, 200)
        ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), color="black", lw=1.5)
        ax.set_xlim(-3, 3)
        ax.set_title(f"proj {i + 1}", fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig("pics/lejepa_projections.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("saved pics/lejepa_projections.png")


if __name__ == "__main__":
    save_shapes_grid()
    save_projections()
