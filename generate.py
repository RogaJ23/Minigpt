"""Sample text from a trained checkpoint.

Usage:
    python generate.py                                   # GPT, 500 chars
    python generate.py --model bigram
    python generate.py --prompt "ROMEO:" --temperature 0.8 --top-k 20
"""

import argparse
from pathlib import Path

import torch

from minigpt.checkpoint import load_checkpoint
from minigpt.tokenizer import CharTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["bigram", "gpt"], default="gpt")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ckpt_dir = Path(args.ckpt_dir)
    tok = CharTokenizer.load(ckpt_dir / "tokenizer.json")
    model, ckpt = load_checkpoint(ckpt_dir / f"{args.model}.pt")
    model.eval()

    unknown = sorted(set(args.prompt) - set(tok.chars))
    if unknown:
        raise SystemExit(f"prompt has characters not in the vocab: {unknown}")
    start = torch.tensor([tok.encode(args.prompt)], dtype=torch.long)
    out = model.generate(start, args.tokens, args.temperature, args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
