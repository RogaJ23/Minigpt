"""Train a character-level language model on tiny Shakespeare.

Usage:
    python train.py --model gpt                  # transformer (Week 2)
    python train.py --model bigram               # Week 1 baseline
    python train.py --model gpt --max-iters 500  # quick smoke test
    python train.py --model gpt --n-layer 6 --n-embd 192 --block-size 256
"""

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from minigpt.checkpoint import build_model, save_checkpoint
from minigpt.data import DATA_PATH, download_data, get_batch, split_data
from minigpt.gpt import GPTConfig
from minigpt.tokenizer import CharTokenizer

# Per-model defaults; any flag you pass on the command line overrides these.
MODEL_DEFAULTS = {
    "bigram": dict(block_size=8, lr=1e-2, max_iters=3000, warmup_iters=0,
                   weight_decay=0.0, dropout=0.0),
    "gpt": dict(block_size=128, lr=1e-3, max_iters=3000, warmup_iters=100,
                weight_decay=0.1, dropout=0.1),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["bigram", "gpt"], default="gpt")
    p.add_argument("--data", default=str(DATA_PATH))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--max-iters", type=int, default=None)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=100)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--warmup-iters", type=int, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # Transformer size (ignored for bigram)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--device", default=None, help="cuda, mps or cpu (auto by default)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default="checkpoints")
    args = p.parse_args()
    for key, value in MODEL_DEFAULTS[args.model].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # Apple Silicon GPU
        return "mps"
    return "cpu"


def get_lr(it, args):
    """Linear warmup, then cosine decay down to 10% of the peak learning rate."""
    if it < args.warmup_iters:
        return args.lr * (it + 1) / args.warmup_iters
    progress = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    min_lr = 0.1 * args.lr
    return min_lr + 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))) * (args.lr - min_lr)


def make_optimizer(model, args):
    """AdamW with weight decay on matrices only (not biases or LayerNorm)."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.99))


@torch.no_grad()
def estimate_loss(model, splits, args, device):
    """Average loss over several random batches per split (less noisy)."""
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
    device = args.device or pick_device()
    print(f"device: {device} | model: {args.model}")

    path = download_data(args.data)
    text = Path(path).read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_data, val_data = split_data(ids)
    splits = {"train": train_data, "val": val_data}
    print(f"chars: {len(text):,} | vocab: {tok.vocab_size} | "
          f"train: {len(train_data):,} | val: {len(val_data):,}")

    if args.model == "bigram":
        config = {"vocab_size": tok.vocab_size}
    else:
        config = asdict(GPTConfig(
            vocab_size=tok.vocab_size, block_size=args.block_size,
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            dropout=args.dropout,
        ))
    model = build_model(args.model, config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")
    opt = make_optimizer(model, args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    tok.save(out_dir / "tokenizer.json")
    history = []

    t0 = time.time()
    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(model, splits, args, device)
            history.append({"iter": it, **losses})
            print(f"iter {it:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | lr {get_lr(it, args):.2e} | "
                  f"{time.time() - t0:.1f}s")
        if it == args.max_iters:
            break

        lr = get_lr(it, args)
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

    ckpt_path = out_dir / f"{args.model}.pt"
    save_checkpoint(ckpt_path, args.model, config, model,
                    iter=args.max_iters, val_loss=history[-1]["val"])
    hist_path = out_dir / f"loss_history_{args.model}.json"
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"saved {ckpt_path} and {hist_path} | total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
