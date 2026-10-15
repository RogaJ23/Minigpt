import math

import torch

from minigpt.data import get_batch, split_data
from minigpt.model import BigramLanguageModel
from minigpt.tokenizer import CharTokenizer


def test_tokenizer_roundtrip():
    text = "hello, world\n"
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text
    assert tok.vocab_size == len(set(text))


def test_tokenizer_save_load(tmp_path):
    tok = CharTokenizer.from_text("abcabc")
    tok.save(tmp_path / "t.json")
    tok2 = CharTokenizer.load(tmp_path / "t.json")
    assert tok2.encode("abc") == tok.encode("abc")


def test_split_sizes():
    train, val = split_data(torch.arange(100), 0.9)
    assert len(train) == 90 and len(val) == 10


def test_batch_shapes_and_shift():
    data = torch.arange(1000)
    x, y = get_batch(data, batch_size=4, block_size=8)
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert torch.equal(y, x + 1)  # targets are inputs shifted by one


def test_initial_loss_near_uniform():
    # An untrained model should predict roughly uniformly: loss ~ ln(vocab).
    torch.manual_seed(0)
    V = 65
    model = BigramLanguageModel(V)
    x = torch.randint(V, (64, 8))
    y = torch.randint(V, (64, 8))
    _, loss = model(x, y)
    assert abs(loss.item() - math.log(V)) < 1.0


def test_generate_length_and_range():
    model = BigramLanguageModel(10)
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), 20, top_k=3)
    assert out.shape == (1, 21)
    assert out.min() >= 0 and out.max() < 10


def test_can_overfit_tiny_data():
    # Sanity check that gradients flow: loss should drop on a repeating pattern.
    torch.manual_seed(0)
    data = torch.tensor([0, 1, 2, 3] * 50)
    model = BigramLanguageModel(4)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    x, y = get_batch(data, 16, 4)
    _, first = model(x, y)
    for _ in range(100):
        x, y = get_batch(data, 16, 4)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first.item() * 0.2
