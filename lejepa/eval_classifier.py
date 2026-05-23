import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from lejepa.encoder import Encoder
from lejepa.fig_gen import make_square_image, make_circle_image, make_triangle_image

MAKERS = [make_square_image, make_circle_image, make_triangle_image]
LABELS = ["Square", "Circle", "Triangle"]
N_SAMPLES = 10

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("data/lejepa_classifier_final.pt", map_location=device)
    encoder = Encoder().to(device)
    head = nn.Linear(64, 3).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])
    encoder.eval()
    head.eval()

    fig, axes = plt.subplots(3, N_SAMPLES, figsize=(N_SAMPLES * 2, 6))
    with torch.no_grad():
        for row, (maker, true_label) in enumerate(zip(MAKERS, LABELS)):
            imgs = torch.stack([maker() for _ in range(N_SAMPLES)]).to(device)
            z = encoder(imgs)
            preds = head(z).argmax(dim=1)
            for col in range(N_SAMPLES):
                pred_label = LABELS[preds[col].item()]
                color = "green" if pred_label == true_label else "red"
                ax = axes[row, col]
                ax.imshow(z[col].cpu().reshape(8, 8).numpy(), cmap="gray")
                ax.set_title(f"true: {true_label}\npred: {pred_label}", fontsize=7, color=color)
                ax.axis("off")
    plt.tight_layout()
    plt.show()
