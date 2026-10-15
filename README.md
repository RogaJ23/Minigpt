# mini-gpt

A small GPT-style language model written from scratch in PyTorch, then profiled
and optimized (mixed precision, `torch.compile`, KV cache, and a custom kernel).

**Status:** Week 1 of 6 — tokenizer, data pipeline, bigram baseline, training loop.

## Quick start (Linux/macOS/WSL)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -q                    # run the tests
python train.py              # trains the bigram baseline (downloads tiny Shakespeare)
python generate.py --tokens 400
python plot_loss.py          # writes loss_curve.png
```

## Layout

```
minigpt/
  tokenizer.py   character-level tokenizer
  data.py        download, train/val split, random batches
  model.py       bigram baseline (transformer arrives in week 2)
train.py         training loop with periodic train/val evaluation
generate.py      sampling with temperature and top-k
plot_loss.py     loss curve plot
tests/           unit tests
LOG.md           running notes
```

## Results

| Model | Params | Val loss |
|-------|--------|----------|
| Uniform guess (ln 65) | – | 4.17 |
| Bigram (week 1) | 4,225 | _fill in_ |
| Transformer (week 2) | _TBD_ | _TBD_ |

## Roadmap

1. Bigram baseline and training loop (this week)
2. Transformer: attention, blocks, full GPT
3. Real dataset (Python code), BPE, experiments
4. Profiling and optimization
5. KV cache and a Triton kernel
6. Roofline analysis, polish
