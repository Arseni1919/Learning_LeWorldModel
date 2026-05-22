import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from lejepa.fig_gen import make_square_image


class Normalize(nn.Module):
    def forward(self, x):
        return x * 2 - 1


class Encoder(nn.Module):
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
        self.mlp = nn.Sequential(
            nn.Linear(24 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


if __name__ == "__main__":
    model = Encoder()
    image = make_square_image().unsqueeze(0)
    with torch.no_grad():
        output = model(image)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(image.squeeze(0).permute(1, 2, 0).numpy())
    axes[0].set_title("Input (3x64x64)")
    axes[0].axis("off")
    axes[1].imshow(output.squeeze().numpy(), cmap="gray")
    axes[1].set_title("Output (1x64x64)")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()
