import torch
import matplotlib.pyplot as plt
from lejepa.encoder import Encoder
from lejepa.fig_gen import make_square_image, make_circle_image, make_triangle_image

MAKERS = [make_square_image, make_circle_image, make_triangle_image]
LABELS = ["Square", "Circle", "Triangle"]
N_SAMPLES = 10

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(latent_dim=3).to(device)
    encoder.load_state_dict(torch.load("data/lejepa_classifier_final.pt", map_location=device)["encoder"])
    encoder.eval()

    activations = {}
    encoder.mlp[3].register_forward_hook(lambda m, i, o: activations.update({"mid": o.detach()}))

    fig, axes = plt.subplots(3, N_SAMPLES, figsize=(N_SAMPLES * 2, 6))
    with torch.no_grad():
        for row, (maker, true_label) in enumerate(zip(MAKERS, LABELS)):
            imgs = torch.stack([maker() for _ in range(N_SAMPLES)]).to(device)
            preds = encoder(imgs).argmax(dim=1)
            mid = activations["mid"].cpu().reshape(N_SAMPLES, 8, 8)
            for col in range(N_SAMPLES):
                pred_label = LABELS[preds[col].item()]
                color = "green" if pred_label == true_label else "red"
                ax = axes[row, col]
                ax.imshow(mid[col].numpy(), cmap="gray")
                ax.set_title(f"true: {true_label}\npred: {pred_label}", fontsize=7, color=color)
                ax.axis("off")
    plt.tight_layout()
    plt.show()
