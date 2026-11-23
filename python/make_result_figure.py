"""
make_result_figure.py - NMSE vs input period ratio T (= 1/m), NARMA-2 and NARMA-5,
with the NARMA target driven by the Arduino servo pulse. Recreates the results
figure of the PHRESCO_2026 README (images/crumpled_paper_result.png).

One recording per multiplier m (the same recordings as the original figure).
motion_tracking_25_3freq.mat is left out: it was recorded at m = 0.50, not 0.25.
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from crumpled_paper_narma import benchmark, narma
from servo_command import DATA_DIR, commanded_input, load_sync, load_tracking

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["motion_tracking_%d_3freq.mat" % p for p in (30, 35, 42, 46, 50, 55, 60, 65, 70, 76, 83, 91, 100)]
sync = load_sync()
T, n2, n5 = [], [], []
for f in FILES:
    tr = load_tracking(os.path.join(DATA_DIR, f))
    n = len(tr.frameNumbers)
    S = tr.blueXPos.reshape(n, -1, order="F")
    u = commanded_input(tr, sync[f])
    T.append(1 / sync[f]["m"])
    n2.append(benchmark(S, narma(u, 2))["NMSE"])
    n5.append(benchmark(S, narma(u, 5))["NMSE"])
o = np.argsort(T)
T, n2, n5 = np.array(T)[o], np.array(n2)[o], np.array(n5)[o]
for t, a, b in zip(T, n2, n5):
    print(f"T = {t:5.2f}   NARMA-2 {a:.3f}   NARMA-5 {b:.3f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(T, n2, "-o", color="b", lw=2.5, ms=9, label="NARMA-2")
ax.plot(T, n5, "-s", color="r", lw=2.5, ms=9, label="NARMA-5")
ax.axhline(0.2, color="0.5", ls="--", lw=1.5, label="20% NMSE Threshold")
ax.set_xscale("log")
ax.set_xticks([1, 1.5, 2, 2.5, 3, 3.5])
ax.set_xticklabels(["1", "1.5", "2", "2.5", "3", "3.5"])
ax.minorticks_off()
ax.set_ylim(0, 0.5)
ax.grid(True, alpha=0.4)
ax.set_xlabel("Input Period ratio T", fontweight="bold", fontsize=12)
ax.set_ylabel("NMSE (Normalized Mean Square Error)", fontweight="bold", fontsize=12)
ax.set_title("NARMA Benchmark Performance vs Input Period\n(input: Arduino servo command)", fontweight="bold", fontsize=13)
ax.legend(loc="upper right")
fig.tight_layout()
out = os.path.join(HERE, "..", "images", "crumpled_paper_result.png")
fig.savefig(out, dpi=200)
print("wrote", out)
