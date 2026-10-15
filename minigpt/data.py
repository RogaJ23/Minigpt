"""Dataset download, train/val split, and batch sampling."""

import urllib.request
from pathlib import Path

import torch

DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DATA_PATH = Path("data/input.txt")


def download_data(path=DATA_PATH, url=DATA_URL):
    """Download tiny Shakespeare (~1 MB) if it isn't already on disk."""
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)
    return path


def split_data(ids, train_frac=0.9):
    """Split a 1-D LongTensor of token ids into train/val tensors."""
    n = int(train_frac * len(ids))
    return ids[:n], ids[n:]


def get_batch(data, batch_size, block_size, device="cpu"):
    """Sample a random batch of (x, y) where y is x shifted by one token.

    x: (batch_size, block_size) input token ids
    y: (batch_size, block_size) next-token targets
    """
    if len(data) <= block_size:
        raise ValueError("data is shorter than block_size + 1")
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
