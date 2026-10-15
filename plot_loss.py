"""Plot train/val loss from checkpoints/loss_history.json -> loss_curve.png"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = Path(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/loss_history.json")
hist = json.loads(path.read_text())
iters = [h["iter"] for h in hist]

plt.figure(figsize=(6, 4))
plt.plot(iters, [h["train"] for h in hist], label="train")
plt.plot(iters, [h["val"] for h in hist], label="val")
plt.xlabel("iteration")
plt.ylabel("cross-entropy loss")
plt.title("Bigram baseline")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
print("wrote loss_curve.png")
