"""Week 2: a GPT-style decoder-only transformer, written from scratch.

Architecture (GPT-2 style, pre-LayerNorm):
    token + position embeddings -> n_layer x Block -> LayerNorm -> logits
    Block = x + Attention(LN(x)), then x + MLP(LN(x))

Shape notation used in comments:
    B = batch size, T = sequence length (<= block_size),
    C = n_embd (embedding size), nh = n_head, hs = C // nh (head size),
    V = vocab_size
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128   # max context length
    n_layer: int = 4        # number of transformer blocks
    n_head: int = 4         # attention heads per block
    n_embd: int = 128       # embedding size (must divide by n_head)
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each position only sees earlier ones."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # One linear layer produces queries, keys and values together: C -> 3C.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection after the heads are concatenated: C -> C.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Lower-triangular matrix of ones: mask[i, j] == 1 when j <= i.
        # Shape (1, 1, block_size, block_size) so it broadcasts over B and nh.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer(
            "mask", mask.view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(self, x):
        B, T, C = x.shape
        hs = C // self.n_head

        # 1. Project to queries, keys and values in one matmul, then split.
        #    Each is (B, T, C).
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 2. Split C into nh heads of size hs, and move the head dim forward so
        #    every head attends independently: (B, T, C) -> (B, nh, T, hs).
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # 3. Attention scores: how much position i should look at position j.
        #    Dividing by sqrt(hs) keeps the scores' variance near 1, so softmax
        #    doesn't saturate into a near one-hot distribution at init.
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)          # (B, nh, T, T)

        # 4. Causal mask: position i may not look at any j > i (the future).
        #    -inf becomes exactly 0 after softmax.
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # 5. Turn scores into weights that sum to 1 over the keys.
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 6. Weighted average of the values.
        y = att @ v                                              # (B, nh, T, hs)

        # 7. Concatenate the heads back together: (B, nh, T, hs) -> (B, T, C).
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Mix information across heads with the output projection.
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network: C -> 4C -> C."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Attention moves information between positions; the MLP then
        # processes each position on its own, with a 4x wider hidden layer.
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    """One transformer block (pre-LayerNorm, like GPT-2)."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Residual connections ("x + ..."): each sub-layer only adds a change
        # to x, and gradients flow straight back through the "+" to early
        # layers, which is what lets deep stacks train. Pre-LayerNorm (normalize
        # before each sub-layer, as in GPT-2) keeps training stable.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)  # what token
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)  # where it is
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: the output layer reuses the token embedding matrix.
        self.lm_head.weight = self.tok_emb.weight
        # Small random init (std 0.02), as in GPT-2. Residual output layers get
        # an extra 1/sqrt(2 * n_layer) so the residual stream doesn't grow with depth.
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"sequence length {T} > block_size {self.config.block_size}"
        )
        # Token embedding says WHAT each token is; position embedding says
        # WHERE it is. Attention by itself has no sense of order, so we add them.
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))    # (B, T, C)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)                                 # (B, T, V)

        loss = None
        if targets is not None:
            # Every position predicts its next token, so all B*T positions
            # are training examples at once.
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively extend idx (B, T) by max_new_tokens tokens."""
        for _ in range(max_new_tokens):
            # Unlike the bigram model, we must crop to the last block_size
            # tokens, because the position table only has block_size rows.
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx
