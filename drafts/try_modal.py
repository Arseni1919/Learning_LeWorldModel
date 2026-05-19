import modal

app = modal.App("gpu-demo")

image = modal.Image.debian_slim().pip_install("torch", "numpy", extra_options="--index-url https://download.pytorch.org/whl/cu121")


@app.function(gpu="t4", image=image)
def square_on_gpu(numbers: list[float]) -> list[float]:
    import torch

    tensor = torch.tensor(numbers, device="cuda")
    print(f"Running on: {torch.cuda.get_device_name(0)}")
    result = tensor ** 2
    return result.tolist()


@app.local_entrypoint()
def main():
    numbers = [1.0, 2.0, 3.0, 4.0, 5.0]
    print(f"Input:  {numbers}")
    results = square_on_gpu.remote(numbers)
    print(f"Squared on GPU: {results}")
