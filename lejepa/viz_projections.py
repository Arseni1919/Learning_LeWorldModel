import torch
import numpy as np
import matplotlib.pyplot as plt
from lejepa.encoder import Encoder
from lejepa.fig_gen import make_square_image, make_circle_image, make_triangle_image

MAKERS = [make_square_image, make_circle_image, make_triangle_image]
LABELS = ["Square", "Circle", "Triangle"]
COLORS = ["steelblue", "tomato", "seagreen"]
N_SAMPLES = 500
M = 16

device = torch.device("cpu")
encoder = Encoder()
ckpt = torch.load("data/lejepa_checkpoint_final.pt", map_location=device)
encoder.load_state_dict(ckpt["encoder"])
encoder.eval()

imgs_by_class = [[maker() for _ in range(N_SAMPLES)] for maker in MAKERS]

with torch.no_grad():
    Z_by_class = [encoder(torch.stack(imgs)).flatten(1) for imgs in imgs_by_class]

Z = torch.cat(Z_by_class, dim=0)
latent_dim = Z.shape[1]

torch.manual_seed(0)
U = torch.nn.functional.normalize(torch.randn(latent_dim, M), dim=0)

fig, axes = plt.subplots(4, 4, figsize=(7, 7))
for i, ax in enumerate(axes.flat):
    for Z_cls, label, color in zip(Z_by_class, LABELS, COLORS):
        H_cls = (Z_cls @ U[:, i]).numpy()
        ax.hist(H_cls, bins=40, density=True, alpha=0.5, color=color, label=label)
    x = np.linspace(-3, 3, 200)
    ax.plot(x, np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi), color="black", lw=1.5, label="N(0,1)")
    ax.set_xlim(-3, 3)
    ax.set_title(f"projection {i + 1}")
    if i == 0:
        ax.legend(fontsize=7)

plt.tight_layout()
plt.show()
