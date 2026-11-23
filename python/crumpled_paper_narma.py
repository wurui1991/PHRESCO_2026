"""
crumpled_paper_narma.py  -  PHRESCO crumpled paper example (NARMA benchmark)
Morphological Computation Group, University of Bristol

    input  u(t) : the signal sent to the servo (Arduino command)      <-- changed
    system      : servo + servo/paper coupling + crumpled paper
    states x(t) : X, Y of the 11 blue markers (22 channels), from video
    readout     : linear least squares, y_hat = W [x; 1]
    target y(t) : NARMA-n driven by u(t)

Until Nov 2026 the example built the NARMA target from the camera-tracked
green marker (a point on the paper next to the servo). That point is already
an output of the physical system, so the servo dynamics and the servo/paper
coupling were left outside the benchmark. Here the target is computed from the
command, so everything between the Arduino and the blue markers is part of the
reservoir. See the repository README.

Usage
    python crumpled_paper_narma.py                                   # default recording, NARMA-2
    python crumpled_paper_narma.py motion_tracking_50_3freq_r2.mat --order 5
    python crumpled_paper_narma.py --input marker                    # old (legacy) definition, for comparison
"""
import argparse
import os

import numpy as np
from scipy.interpolate import PchipInterpolator

from servo_command import DATA_DIR, commanded_input, load_sync, load_tracking


def narma(u, order=2):
    """NARMA-n target from input u (u is rescaled to [0, 0.5], the standard NARMA range)."""
    u = np.asarray(u, float)
    u = u - u.min()
    u = 0.5 * u / u.max()
    n_s = len(u)
    y = np.zeros(n_s)
    if order == 2:
        for k in range(1, n_s - 1):
            y[k + 1] = 0.4 * y[k] + 0.4 * y[k] * y[k - 1] + 0.6 * u[k] ** 3 + 0.1
    else:
        n = order
        for k in range(n - 1, n_s - 1):
            y[k + 1] = 0.3 * y[k] + 0.05 * y[k] * y[k - n + 1:k + 1].sum() + 1.5 * u[k - n + 1] * u[k] + 0.1
    if np.abs(y).max() > 10:
        y = y / np.abs(y).max()
    return y


def benchmark(states, target, train_fraction=0.8, washout_fraction=0.2, interp_factor=1):
    """Same pre-processing and split as CrumpledPaper.m. Returns dict with NMSE and the test traces."""
    n = len(target)
    if interp_factor > 1:
        t_new = np.linspace(0, n - 1, n * interp_factor)
        states = PchipInterpolator(np.arange(n), states)(t_new)
        target = PchipInterpolator(np.arange(n), target)(t_new)
        n = len(t_new)
    s = 2 * (states - states.min(0)) / (states.max(0) - states.min(0)) - 1
    s = s - s.mean(0)
    y = 2 * (target - target.min()) / (target.max() - target.min()) - 1
    y = y - y.mean()
    X = np.c_[s, np.ones(n)]
    valid = np.arange(round(washout_fraction * n), n)
    n_train = round(train_fraction * len(valid))
    tr, te = valid[:n_train], valid[n_train:]
    W, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
    y_hat = X[te] @ W
    nmse = np.mean((y_hat - y[te]) ** 2) / np.var(y[te])
    return dict(NMSE=float(nmse), W=W, test_idx=te, y_test=y[te], y_pred=y_hat, y_all=y)


def run(mat_file, order=2, input_source="command", interp_factor=1, plot=True):
    tr = load_tracking(os.path.join(DATA_DIR, mat_file))
    n = len(np.atleast_1d(tr.frameNumbers))
    fs = float(tr.samplerate)
    states = tr.blueXPos.reshape(n, -1, order="F")  # [frames x 22], same order as MATLAB reshape
    if input_source == "command":
        u = commanded_input(tr, load_sync()[os.path.basename(mat_file)])
    elif input_source == "marker":  # legacy definition: x-coordinate of the green paper marker
        u = tr.servoPos[:, 0]
    else:
        raise ValueError(input_source)
    res = benchmark(states, narma(u, order), interp_factor=interp_factor)
    res["u"] = u
    print(f"{mat_file}  NARMA-{order}  input={input_source:7s}  NMSE = {res['NMSE']:.4f}")

    if plot:
        import matplotlib.pyplot as plt
        t = np.arange(n) / fs
        if interp_factor > 1:
            t = np.linspace(t[0], t[-1], n * interp_factor)
        fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=False)
        ax[0].plot(np.arange(n) / fs, u, "k", lw=0.8)
        ax[0].set_ylabel("servo command" if input_source == "command" else "green marker x (px)")
        ax[0].set_title(f"Input u(t)  [{input_source}]")
        ax[1].plot(np.arange(n) / fs, states - states.mean(0), lw=0.5)
        ax[1].set_ylabel("blue markers (px, centred)")
        tt = t[res["test_idx"]]
        ax[2].plot(tt, res["y_test"], "b", lw=1.5, label="NARMA target")
        ax[2].plot(tt, res["y_pred"], "r--", lw=1.2, label="linear readout")
        ax[2].set_title(f"NARMA-{order} on test set, NMSE = {res['NMSE']:.3f}")
        ax[2].set_xlabel("time (s)")
        ax[2].legend(loc="upper right")
        fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f"result_{os.path.splitext(os.path.basename(mat_file))[0]}_NARMA{order}_{input_source}.png")
        fig.savefig(out, dpi=130)
        print("figure:", out)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mat_file", nargs="?", default="motion_tracking_50_3freq.mat")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--input", dest="input_source", choices=["command", "marker"], default="command")
    ap.add_argument("--interp", type=int, default=1)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args()
    run(a.mat_file, a.order, a.input_source, a.interp, plot=not a.no_plot)
