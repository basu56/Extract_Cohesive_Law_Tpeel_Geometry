"""Synthetic data for the inversions: ABAQUS P-delta records, observed with the D29 bin operator.  I1.

Units: mm, N.  The data come from a different solver (ABAQUS) than the model (tslid.fe); the mesh is
the same node for node (I1c adds a finer-mesh truth).
"""
import os

import numpy as np

from tslid.analysis.compare_ld import load
from tslid.fe.observation import DELTA_BIN, observe, windows


def abaqus_record(path):
    """ABAQUS record <rundir>/<job>: dict(delta, P) starting at (0, 0), d_end (compare_ld window: the
    sigma_c front reaches L - 15 mm), delta_peak, P_peak, params (deck metadata)."""
    rd, job = os.path.split(path)
    x = load(rd, job)
    d, P = np.asarray(x["d"], float), np.asarray(x["P"], float)
    if d[0] > 0:
        d, P = np.r_[0.0, d], np.r_[0.0, P]
    i = int(np.argmax(P))
    return dict(job=job, delta=d, P=P, d_end=float(x["d_end"]), delta_peak=float(d[i]), P_peak=float(P[i]),
                params=x["p"])


def observed(record, width=DELTA_BIN):
    """{window: (edges, y)} for the windows "prepeak" (0 .. delta_peak) and "full" (0 .. d_end)."""
    W = windows(record["d_end"], record["delta_peak"], width)
    return {w: (e, observe(record["delta"], record["P"], e)) for w, e in W.items()}


def add_noise(y, s, seed):
    """y + s * N(0, 1), independent per bin, with a recorded seed (noise levels are assumed, D31)."""
    return np.asarray(y) + s * np.random.default_rng(seed).standard_normal(np.shape(y))
