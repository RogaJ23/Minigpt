"""Character-level tokenizer.

Maps every unique character in the training text to an integer id.
In week 3 we replace this with a BPE tokenizer.
"""

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(text)

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def save(self, path):
        Path(path).write_text(json.dumps(self.chars))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))
