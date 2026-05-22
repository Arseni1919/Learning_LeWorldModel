import math
import torch
import matplotlib.pyplot as plt


def random_distinct_colors(n, min_dist=0.4):
    colors = []
    while len(colors) < n:
        c = torch.rand(3)
        if all(torch.norm(c - existing) >= min_dist for existing in colors):
            colors.append(c)
    return colors


def random_center(max_offset=10):
    offset = torch.randint(-max_offset, max_offset + 1, (2,))
    return 32 + offset[0].item(), 32 + offset[1].item()


def max_fit(cx, cy, canvas=64):
    return min(cx, canvas - cx, cy, canvas - cy)


def make_square_image():
    bg, fg = random_distinct_colors(2)
    image = bg[:, None, None].expand(3, 64, 64).clone()
    cx, cy = random_center()
    half = torch.randint(8, min(16, max_fit(cx, cy)) + 1, (1,)).item()
    image[:, cy - half:cy + half, cx - half:cx + half] = fg[:, None, None]
    return image


def make_circle_image():
    bg, fg = random_distinct_colors(2)
    image = bg[:, None, None].expand(3, 64, 64).clone()
    cx, cy = random_center()
    radius = torch.randint(10, min(18, max_fit(cx, cy)) + 1, (1,)).item()
    y, x = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    image[:, mask] = fg[:, None]
    return image


def make_triangle_image():
    bg, fg = random_distinct_colors(2)
    image = bg[:, None, None].expand(3, 64, 64).clone()
    cx, cy = random_center()
    R = torch.randint(12, min(20, max_fit(cx, cy)) + 1, (1,)).item()
    y, x = torch.meshgrid(torch.arange(64, dtype=torch.float32), torch.arange(64, dtype=torch.float32), indexing="ij")
    angles = [-math.pi / 2, -math.pi / 2 + 2 * math.pi / 3, -math.pi / 2 + 4 * math.pi / 3]
    v0 = (cx + R * math.cos(angles[0]), cy + R * math.sin(angles[0]))
    v1 = (cx + R * math.cos(angles[1]), cy + R * math.sin(angles[1]))
    v2 = (cx + R * math.cos(angles[2]), cy + R * math.sin(angles[2]))
    def cross(ax, ay, bx, by, px, py):
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    d0 = cross(v0[0], v0[1], v1[0], v1[1], x, y)
    d1 = cross(v1[0], v1[1], v2[0], v2[1], x, y)
    d2 = cross(v2[0], v2[1], v0[0], v0[1], x, y)
    mask = ((d0 >= 0) & (d1 >= 0) & (d2 >= 0)) | ((d0 <= 0) & (d1 <= 0) & (d2 <= 0))
    image[:, mask] = fg[:, None]
    return image


def show_grid(images, title):
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    fig.suptitle(title)
    for ax, img in zip(axes.flat, images):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    squares = [make_square_image() for _ in range(10)]
    circles = [make_circle_image() for _ in range(10)]
    triangles = [make_triangle_image() for _ in range(10)]
    show_grid(squares, "Squares")
    show_grid(circles, "Circles")
    show_grid(triangles, "Triangles")
