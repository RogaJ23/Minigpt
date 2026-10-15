"""Train the bigram baseline on tiny Shakespeare.

Usage:
    python train.py                      # defaults
    python train.py --max-iters 5000 --lr 3e-3
"""

import argparse
import json
import time
from pathlib import Path

import torch

from minigpt.data import DATA_PATH, download_data, get_batch, split_data
from minigpt.model import BigramLanguageModel
from minigpt.tokenizer import CharTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DATA_PATH))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=8)
    p.add_argument("--max-iters", type=int, default=3000)
    p.add_argument("--eval-interval", type=int, default=300)
    p.add_argument("--eval-iters", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default="checkpoints")
    return p.parse_args()


@torch.no_grad()
def estimate_loss(model, splits, args, device):
    """Average loss over several random batches for each split (less noisy)."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(data, args.batch_size, args.block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    path = download_data(args.data)
    text = Path(path).read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_data, val_data = split_data(ids)
    splits = {"train": train_data, "val": val_data}
    print(f"chars: {len(text):,} | vocab: {tok.vocab_size} | "
          f"train: {len(train_data):,} | val: {len(val_data):,}")

    model = BigramLanguageModel(tok.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    tok.save(out_dir / "tokenizer.json")
    history = []

    t0 = time.time()
    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0:
            losses = estimate_loss(model, splits, args, device)
            history.append({"iter": it, **losses})
            print(f"iter {it:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | {time.time() - t0:.1f}s")
        if it == args.max_iters:
            break
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    torch.save(model.state_dict(), out_dir / "bigram.pt")
    (out_dir / "loss_history.json").write_text(json.dumps(history, indent=2))
    print(f"saved model and loss history to {out_dir}/")


if __name__ == "__main__":
    main()
