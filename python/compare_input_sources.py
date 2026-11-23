"""
compare_input_sources.py  -  old vs new NARMA input definition on every recording.

  old    : u = x-coordinate of the green marker on the paper (what CrumpledPaper.m used until Nov 2026)
  new    : u = Arduino servo command, synchronised to the video (servo_command.py)
  checks : the same with the command sign flipped, and with the command shifted by
           +/- 2 frames (33 ms), to show how much the result depends on the two
           things the legacy recordings could not pin down.

Writes comparison_NMSE.csv and comparison_NMSE.png in this folder.
"""
import csv
import os

import numpy as np

from crumpled_paper_narma import benchmark, narma
from servo_command import DATA_DIR, arduino_pulse_us, load_sync, load_tracking

HERE = os.path.dirname(os.path.abspath(__file__))
sync = load_sync()
rows = []
for f, e in sorted(sync.items(), key=lambda kv: (kv[1]["m"], kv[0])):
    if f == "motion_tracking_35_3freq1.mat":  # same tracking data as motion_tracking_35_3freq.mat
        continue
    tr = load_tracking(os.path.join(DATA_DIR, f))
    n = len(tr.frameNumbers)
    fs = float(tr.samplerate)
    k = np.arange(n) / fs
    S = tr.blueXPos.reshape(n, -1, order="F")

    def cmd(shift_frames=0, sign=1.0):
        return arduino_pulse_us(e["clock_scale"] * (e["t0"] + k + shift_frames / fs), e["m"], sign=sign * e["sign"])

    r = dict(file=f, m=e["m"], sync_R2=round(e["fit_R2"], 3))
    for order in (2, 5):
        r[f"old_N{order}"] = benchmark(S, narma(tr.servoPos[:, 0], order))["NMSE"]
        r[f"new_N{order}"] = benchmark(S, narma(cmd(), order))["NMSE"]
        r[f"new_signflip_N{order}"] = benchmark(S, narma(cmd(sign=-1), order))["NMSE"]
    r["new_N2_cmd_2frames_earlier"] = benchmark(S, narma(cmd(+2), 2))["NMSE"]  # command leads marker (servo latency)
    r["new_N2_cmd_2frames_later"] = benchmark(S, narma(cmd(-2), 2))["NMSE"]
    r["corr_cmd_vs_old_input"] = float(np.corrcoef(cmd(), tr.servoPos[:, 0])[0, 1])
    rows.append(r)

keys = list(rows[0].keys())
with open(os.path.join(HERE, "comparison_NMSE.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, keys)
    w.writeheader()
    for r in rows:
        w.writerow({k_: (f"{v:.4f}" if isinstance(v, float) else v) for k_, v in r.items()})

print(f"{'file':34s} {'m':>5s} | {'old N2':>7s} {'new N2':>7s} {'flip':>6s} {'lead2':>6s} {'lag2':>6s} | {'old N5':>7s} {'new N5':>7s}")
for r in rows:
    print(f"{r['file']:34s} {r['m']:5.2f} | {r['old_N2']:7.3f} {r['new_N2']:7.3f} {r['new_signflip_N2']:6.3f} "
          f"{r['new_N2_cmd_2frames_earlier']:6.3f} {r['new_N2_cmd_2frames_later']:6.3f} | {r['old_N5']:7.3f} {r['new_N5']:7.3f}")
for key in ("old_N2", "new_N2", "new_signflip_N2", "old_N5", "new_N5"):
    v = np.array([r[key] for r in rows])
    print(f"{key:18s} median {np.median(v):.3f}   recordings with NMSE < 0.20: {int((v < 0.2).sum())}/{len(v)}")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
x = np.arange(len(rows))
lab = [f"{r['m']:.2f}" + ("*" if "r2" in r["file"] else "") + ("\u2020" if "_25_" in r["file"] else "") for r in rows]
for a_, order in zip(ax, (2, 5)):
    a_.bar(x - 0.2, [r[f"old_N{order}"] for r in rows], 0.4, color="#9aa5b1", label="old: u = green marker x (on paper)")
    a_.bar(x + 0.2, [r[f"new_N{order}"] for r in rows], 0.4, color="#2b6cb0", label="new: u = Arduino servo pulse (us)")
    a_.axhline(0.2, color="#c53030", ls="--", lw=1)
    a_.text(-0.6, 0.205, "PHRESCO threshold 0.20", color="#c53030", ha="left", va="bottom", fontsize=8,
            bbox=dict(fc="white", ec="none", pad=1))
    a_.set_xticks(x)
    a_.set_xticklabels(lab, rotation=60, fontsize=8)
    a_.set_xlabel("frequency multiplier m  (* repeat recording, \u2020 file named 25, recorded at 0.50)")
    a_.set_title(f"NARMA-{order}")
ax[0].set_ylabel("test NMSE (lower is better)")
ax[0].legend(fontsize=8, loc="upper center")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "comparison_NMSE.png"), dpi=140)
print("wrote comparison_NMSE.csv / .png")
