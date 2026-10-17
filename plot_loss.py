"""Plot val (solid) and train (dashed) loss for every trained model -> loss_curve.png"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

paths = sorted(Path("checkpoints").glob("loss_history_*.json"))
if not paths:
    raise SystemExit("no checkpoints/loss_history_*.json found; run train.py first")

plt.figure(figsize=(7, 4.5))
for path in paths:
    name = path.stem.removeprefix("loss_history_")
    hist = json.loads(path.read_text())
    iters = [h["iter"] for h in hist]
    (line,) = plt.plot(iters, [h["val"] for h in hist],
                       label=f"{name} val ({hist[-1]['val']:.3f})")
    plt.plot(iters, [h["train"] for h in hist], "--",
             color=line.get_color(), alpha=0.6, label=f"{name} train")

plt.xlabel("iteration")
plt.ylabel("cross-entropy loss")
plt.title("Character-level loss on tiny Shakespeare")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
print("wrote loss_curve.png")
