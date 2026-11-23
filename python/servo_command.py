"""
servo_command.py  -  PHRESCO crumpled paper example
Morphological Computation Group, University of Bristol

The NARMA input u(t) must be the signal WE SEND to the servo, not a quantity
measured on the paper. This module rebuilds that command, the servo pulse width
in microseconds, exactly as the Arduino computes it, and synchronises it to the video.

Sketch used for the 2025 recordings: Crumpled_paper.ino (repository root)
(EMAX ES9256 II HV servo, waveform 0 = sine):
    sig(t)   = sin(2*pi*f1*m*t) * sin(2*pi*f2*m*t) * sin(2*pi*f3*m*t)      f = 2.11, 3.73, 4.33 Hz
    target   = 800 + int( (sig*ampScale + 1)/2 * 1400 )                   [us], ampScale = 1
    every 5 ms: pulse += clamp(target - pulse, -500, +500); servo.writeMicroseconds(pulse)
    m  = freqMult (the "50" in motion_tracking_50_3freq.mat -> m = 0.50)
    t  = (millis() - t0)/1000, seconds since the Arduino started
The tidied public sketch (Arduino_crumpled_paper.ino, Sep 2025: 1500 + 500*0.5*sig,
20 ms updates) gives the same waveform up to offset and gain, which the NARMA
input scaling removes; use sketch="public" to model it instead.

Why a sync step is needed (legacy recordings only)
    The 2025 recordings did not log the Arduino clock, so the time t0 at which
    the first video frame was taken is unknown. We recover it by fitting the
    command to the green marker motion (two global parameters: t0 and a tiny
    clock-rate factor between the iPhone and the Arduino). A global time shift
    cannot absorb any dynamics: servo bandwidth, saturation, backlash, the way
    the servo horn grips and buckles the paper all stay inside the system.
    What it does absorb is the constant transport latency between command and
    green-marker motion (equivalent to a fixed readout delay).
    NEW RECORDINGS: log the command and a sync event (e.g. an LED the camera
    sees when the Arduino starts) so that this fit is not needed.

Usage
    python servo_command.py                 # fit all motion_tracking_*_3freq*.mat, write servo_command_sync.json
    python servo_command.py file1.mat ...   # fit selected files
"""
import glob
import json
import os
import re
import sys

import numpy as np
import scipy.io as sio
from scipy.optimize import minimize

BASE_FREQS = np.array([2.11, 3.73, 4.33])  # Hz, from the Arduino sketch
SKETCHES = {
    # Crumpled_paper.ino, used for the 2025 recordings
    "lab2025": dict(step=0.005, amp=1.0, us_min=800, us_max=2200, slew=500),
    # Arduino_crumpled_paper.ino, public version on GitHub
    "public": dict(step=0.020, amp=0.5, us_min=1000, us_max=2000, slew=None),
}
SKETCH = "lab2025"
AMPLITUDE = 0.5      # only used by the dimensionless sync fit
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")   # repository root
SYNC_FILE = os.path.join(DATA_DIR, "servo_command_sync.json")


def arduino_signal(t_arduino, m, amplitude=AMPLITUDE, zero_order_hold=True, step=SKETCHES[SKETCH]["step"]):
    """Dimensionless composite sine at Arduino time t (used for the sync fit)."""
    t = np.asarray(t_arduino, dtype=float)
    if zero_order_hold:  # the Arduino only updates the pulse every `step` seconds
        t = np.floor(t / step) * step
    ph = 2 * np.pi * m * t
    return amplitude * np.sin(BASE_FREQS[0] * ph) * np.sin(BASE_FREQS[1] * ph) * np.sin(BASE_FREQS[2] * ph)


def arduino_pulse_us(t_arduino, m, sign=1.0, sketch=SKETCH):
    """Servo pulse width (us) that the Arduino is writing at time t: the actual servo command."""
    cfg = SKETCHES[sketch]
    t = np.asarray(t_arduino, dtype=float)
    # simulate the update loop on the Arduino's own time grid, from slightly before the first frame
    t_grid = np.arange(np.floor(t.min() / cfg["step"]) - 50, np.floor(t.max() / cfg["step"]) + 2) * cfg["step"]
    sig = sign * arduino_signal(t_grid, m, amplitude=1.0, zero_order_hold=False)
    if sketch == "lab2025":
        target = cfg["us_min"] + np.trunc((sig * cfg["amp"] + 1) * 0.5 * (cfg["us_max"] - cfg["us_min"]))
    else:
        target = 1500 + np.trunc(np.clip(sig * cfg["amp"], -1, 1) * (cfg["us_max"] - cfg["us_min"]) / 2)
        target = np.clip(target, cfg["us_min"], cfg["us_max"])
    pulse = target.copy()
    if cfg["slew"] is not None:  # slew-rate limiter (never active with these settings, kept for fidelity)
        for i in range(1, len(pulse)):
            pulse[i] = pulse[i - 1] + np.clip(target[i] - pulse[i - 1], -cfg["slew"], cfg["slew"])
    idx = np.floor(t / cfg["step"]).astype(int) - int(round(t_grid[0] / cfg["step"]))
    return pulse[idx]


def find_recording(mat_path):
    """Accept a path, or a bare file name that lives in the repository root or in data/."""
    for p in (mat_path, os.path.join(DATA_DIR, os.path.basename(mat_path)),
              os.path.join(DATA_DIR, "data", os.path.basename(mat_path))):
        if os.path.exists(p):
            return p
    raise FileNotFoundError(mat_path)


def load_tracking(mat_path):
    return sio.loadmat(find_recording(mat_path), squeeze_me=True, struct_as_record=False)["trackingData"]


def multiplier_from_name(fname):
    """motion_tracking_50_3freq.mat -> 0.50"""
    return int(re.findall(r"motion_tracking_(\d+)_", os.path.basename(fname))[0]) / 100.0


def commanded_input(tracking, sync):
    """Servo pulse width (us) at every video frame, using a sync entry from servo_command_sync.json."""
    n = len(np.atleast_1d(tracking.frameNumbers))
    k = np.arange(n) / float(tracking.samplerate)
    t_arduino = sync["clock_scale"] * (sync["t0"] + k)
    return arduino_pulse_us(t_arduino, sync["m"], sign=sync["sign"])


def fit_sync(mat_path, m=None, t_search=2400.0, min_R2=0.5):
    """Fit the sync. If the file name's multiplier gives a poor fit, try the other multipliers used."""
    e = _fit_sync(mat_path, m, t_search)
    if m is None and e["fit_R2"] < min_R2:
        alts = [_fit_sync(mat_path, mm, t_search) for mm in
                (0.25, 0.3, 0.35, 0.42, 0.46, 0.5, 0.55, 0.6, 0.65, 0.7, 0.76, 0.83, 0.91, 1.0) if mm != e["m"]]
        best = max(alts, key=lambda a: a["fit_R2"])
        if best["fit_R2"] > e["fit_R2"]:
            best["note"] += (f"; WARNING: file name says m={e['m']:.2f} but the recorded motion only fits "
                             f"m={best['m']:.2f} (R2 {e['fit_R2']:.2f} vs {best['fit_R2']:.2f})")
            print("  ", best["note"][best["note"].index("WARNING"):])
            e = best
    return e


def _fit_sync(mat_path, m=None, t_search=2400.0):
    """Find the Arduino time of the first video frame (t0) by matching the command to the green marker."""
    tr = load_tracking(mat_path)
    m = multiplier_from_name(mat_path) if m is None else m
    fs = float(tr.samplerate)
    P = tr.servoPos - tr.servoPos.mean(0)
    n = len(P)
    k = np.arange(n) / fs
    # main direction of the green marker's motion
    _, _, vt = np.linalg.svd(P, full_matrices=False)
    x = P @ vt[0]
    x = (x - x.mean()) / x.std()

    # 1) coarse search over t0. The command repeats exactly every 100/m s
    #    (all base frequencies are multiples of 0.01 Hz), so one period is enough.
    period = 100.0 / m
    best_r, best_t0 = 0.0, 0.0
    for c0 in np.arange(0.0, min(t_search, period), 10.0):
        t0 = c0 + np.arange(0.0, 10.0, 1 / 240)
        ph = 2 * np.pi * m * (t0[:, None] + k[None, :])
        S = np.sin(BASE_FREQS[0] * ph) * np.sin(BASE_FREQS[1] * ph) * np.sin(BASE_FREQS[2] * ph)
        S -= S.mean(1, keepdims=True)
        S /= S.std(1, keepdims=True)
        r = np.abs(S @ x / n)
        i = int(np.argmax(r))
        if r[i] > best_r:
            best_r, best_t0 = float(r[i]), float(t0[i])

    # 2) refine t0 and the clock-rate factor with the full 2-D marker position
    A = np.c_[P, np.ones(n)]

    def neg_r2(p):
        s = arduino_signal(p[1] * (p[0] + k), m, zero_order_hold=False)
        w, *_ = np.linalg.lstsq(A, s, rcond=None)
        return -(1 - np.var(s - A @ w) / np.var(s))

    # several starting clock factors: the iPhone/Arduino clocks differ by up to ~0.5 %
    opt = min((minimize(neg_r2, [best_t0, s0], method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-10))
               for s0 in np.arange(0.996, 1.0081, 0.002)), key=lambda o: o.fun)
    t0, scale = map(float, opt.x)

    # Sign convention. The marker cannot tell us which way the servo turns for a
    # positive command (that is a property of the servo and camera placement).
    # We use: positive command <-> green marker moves towards +x in the image,
    # the same direction the old script used as its input. CHECK THIS on the rig.
    c = arduino_signal(scale * (t0 + k), m)
    sign = float(np.sign(np.corrcoef(c, tr.servoPos[:, 0])[0, 1]))
    return dict(file=os.path.basename(mat_path), m=m, t0=t0, clock_scale=scale, sign=sign,
                fit_R2=float(-opt.fun), coarse_corr=best_r,
                note="t0 = Arduino time (s) at video frame 1; t_arduino = clock_scale*(t0 + frame_time)")


def load_sync(path=SYNC_FILE):
    with open(path) as f:
        return {e["file"]: e for e in json.load(f)}


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(DATA_DIR, "motion_tracking_*_3freq*.mat")) +
                                   glob.glob(os.path.join(DATA_DIR, "data", "motion_tracking_*_3freq*.mat")))
    entries = load_sync() if os.path.exists(SYNC_FILE) else {}
    for f in files:
        f = find_recording(f)
        e = fit_sync(f)
        entries[e["file"]] = e
        print(f"{e['file']:36s} m={e['m']:.2f}  t0={e['t0']:9.3f} s  clock={e['clock_scale']:.5f}  "
              f"sign={e['sign']:+.0f}  fit R2={e['fit_R2']:.3f}", flush=True)
    with open(SYNC_FILE, "w") as fh:
        json.dump(sorted(entries.values(), key=lambda e: e["file"]), fh, indent=1)
    print("wrote", SYNC_FILE)
