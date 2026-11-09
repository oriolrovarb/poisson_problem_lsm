"""Sequential baseline: performance (Mlup/s) vs grid size N, with cache sizes.

Usage (from the repo folder):  python3 plotting/plot_baseline.py results/baseline.csv
Plot is written to plots/<csv name>_performance.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no screen needed, so it works on the cluster
import matplotlib.pyplot as plt

# hpcintro nodes: Xeon E5-2650 v4 (Intel spec, Week02 slides; lscpu in the job output)
CACHES = [                     # (label, bytes available to ONE core, line style)
    ("L1d 32 KiB", 32 * 1024, ":"),
    ("L2 256 KiB", 256 * 1024, "-."),
    ("L3 30 MiB", 30 * 1024**2, "--"),
]
ARRAYS = 3                     # u, u_new, f
BYTES_PER_VALUE = 8            # float64


def n_from_mb(mb):
    """Grid size N whose arrays u, u_new and f together take `mb` MB."""
    return np.cbrt(np.asarray(mb, dtype=float) * 1e6 / (ARRAYS * BYTES_PER_VALUE)) - 2


csv_path = sys.argv[1]
name = os.path.splitext(os.path.basename(csv_path))[0]
os.makedirs("plots", exist_ok=True)

data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
N = np.atleast_1d(data["N"]).astype(int)
mlups = np.atleast_1d(data["mlups"])

# Average the repetitions for each N
n_values = np.unique(N)
mlups_mean = np.array([mlups[N == n].mean() for n in n_values])

fig, ax = plt.subplots()
ax.plot(n_values, mlups_mean, "o-", label="Measured")
# Vertical line where u, u_new and f together no longer fit in each cache level
for label, size, style in CACHES:
    n_cache = n_from_mb(size / 1e6)
    ax.axvline(n_cache, color="k", linestyle=style, label=f"{label} (N ≈ {n_cache:.0f})")
ax.set_xlim(left=0)
ax.set_xlabel("Grid size N (interior points per axis)")
ax.set_ylabel("Performance [Mlup/s]")
ax.set_title("Sequential baseline — Poisson, Jacobi")
ax.grid(True)
ax.legend(loc="upper right")

fig.savefig(f"plots/{name}_performance.png", bbox_inches="tight")
plt.close(fig)
