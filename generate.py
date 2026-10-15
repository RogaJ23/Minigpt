"""Sample text from a trained checkpoint.

Usage:
    python generate.py --tokens 500 --temperature 0.9 --top-k 20
"""

import argparse
from pathlib import Path

import torch

from minigpt.model import BigramLanguageModel
from minigpt.tokenizer import CharTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    ckpt = Path(args.ckpt_dir)
    tok = CharTokenizer.load(ckpt / "tokenizer.json")
    model = BigramLanguageModel(tok.vocab_size)
    model.load_state_dict(torch.load(ckpt / "bigram.pt", map_location="cpu"))
    model.eval()

    start = torch.zeros((1, 1), dtype=torch.long)  # token id 0 (usually "\n")
    out = model.generate(start, args.tokens, args.temperature, args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
