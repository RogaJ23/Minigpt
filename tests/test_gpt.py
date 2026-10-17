"""Tests for the Week 2 transformer. Run: pytest tests/test_gpt.py -v"""

import math

import pytest
import torch
import torch.nn.functional as F

from minigpt.gpt import MLP, GPT, Block, CausalSelfAttention, GPTConfig


def tiny_config(**overrides):
    cfg = dict(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    cfg.update(overrides)
    return GPTConfig(**cfg)


# ---- CausalSelfAttention -------------------------------------------------

def test_attention_output_shape():
    torch.manual_seed(0)
    attn = CausalSelfAttention(tiny_config()).eval()
    assert attn(torch.randn(3, 8, 16)).shape == (3, 8, 16)
    # Sequences shorter than block_size must work too (mask[:, :, :T, :T]).
    assert attn(torch.randn(3, 5, 16)).shape == (3, 5, 16)


def test_attention_matches_pytorch_reference():
    """Your attention should match PyTorch's built-in fused version."""
    torch.manual_seed(0)
    cfg = tiny_config()
    attn = CausalSelfAttention(cfg).eval()
    x = torch.randn(2, 6, cfg.n_embd)
    B, T, C = x.shape
    hs = C // cfg.n_head
    with torch.no_grad():
        q, k, v = attn.c_attn(x).split(C, dim=2)
        q, k, v = (t.view(B, T, cfg.n_head, hs).transpose(1, 2) for t in (q, k, v))
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ref = attn.c_proj(ref.transpose(1, 2).contiguous().view(B, T, C))
        out = attn(x)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


# ---- MLP and Block --------------------------------------------------------

def test_mlp_and_block_shapes():
    torch.manual_seed(0)
    cfg = tiny_config()
    x = torch.randn(2, 8, cfg.n_embd)
    assert MLP(cfg).eval()(x).shape == x.shape
    assert Block(cfg).eval()(x).shape == x.shape


def test_block_has_residual_connection():
    """With the sub-layers' output projections zeroed, a block must be the identity."""
    torch.manual_seed(0)
    cfg = tiny_config()
    block = Block(cfg).eval()
    with torch.no_grad():
        for lin in (block.attn.c_proj, block.mlp.c_proj):
            lin.weight.zero_()
            lin.bias.zero_()
    x = torch.randn(2, 8, cfg.n_embd)
    torch.testing.assert_close(block(x), x)


# ---- GPT ------------------------------------------------------------------

def test_forward_shapes_and_initial_loss():
    torch.manual_seed(0)
    model = GPT(tiny_config(vocab_size=65, n_embd=32))
    idx = torch.randint(65, (4, 8))
    targets = torch.randint(65, (4, 8))
    logits, loss = model(idx, targets)
    assert logits.shape == (4, 8, 65)
    # Untrained model ~ uniform guess, so loss ~ ln(65) ~ 4.17.
    assert abs(loss.item() - math.log(65)) < 0.3
    _, loss = model(idx)
    assert loss is None


def test_no_peeking_at_future_tokens():
    """Changing token 5 must not change predictions at positions 0-4."""
    torch.manual_seed(0)
    model = GPT(tiny_config()).eval()
    idx = torch.randint(11, (1, 8))
    changed = idx.clone()
    changed[0, 5] = (idx[0, 5] + 1) % 11
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(changed)
    torch.testing.assert_close(a[:, :5], b[:, :5])
    assert not torch.allclose(a[:, 5:], b[:, 5:])


def test_rejects_input_longer_than_block_size():
    model = GPT(tiny_config())
    with pytest.raises(AssertionError):
        model(torch.zeros((1, 9), dtype=torch.long))


def test_generate_past_block_size():
    torch.manual_seed(0)
    model = GPT(tiny_config()).eval()
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), 20, top_k=5)
    assert out.shape == (1, 21)
    assert out.min() >= 0 and out.max() < 11


def test_can_overfit_one_batch():
    """If the model can't memorize a single batch, something is wired wrong."""
    torch.manual_seed(0)
    model = GPT(tiny_config())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    idx = torch.randint(11, (4, 8))
    targets = torch.randint(11, (4, 8))
    _, first = model(idx, targets)
    for _ in range(300):
        _, loss = model(idx, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * first.item()
