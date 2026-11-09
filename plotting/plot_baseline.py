"""Sequential baseline: performance (Mlup/s and GB/s) vs grid size N.

Usage (from the repo folder):  python3 plotting/plot_baseline.py results/baseline.csv
Plot is written to plots/<csv name>_performance.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no screen needed, so it works on the cluster
import matplotlib.pyplot as plt

# hpcintro nodes: Xeon E5-2650 v4 (Week02 slides)
L3_BYTES = 30e6                # L3 cache per socket
BYTES_PER_LUP = 64             # same convention as poisson/metrics.py

csv_path = sys.argv[1]
name = os.path.splitext(os.path.basename(csv_path))[0]
os.makedirs("plots", exist_ok=True)

data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
N = np.atleast_1d(data["N"]).astype(int)
mlups = np.atleast_1d(data["mlups"])

# Average the repetitions for each N
n_values = np.unique(N)
mlups_mean = np.array([mlups[N == n].mean() for n in n_values])

# N where u, u_new and f together no longer fit in L3: 3 * (N+2)^3 * 8 bytes
n_l3 = (L3_BYTES / (3 * 8)) ** (1 / 3) - 2

fig, ax = plt.subplots()
ax.plot(n_values, mlups_mean, "o-", label="Measured")
ax.axvline(n_l3, color="k", linestyle="--", label=f"Arrays exceed L3 (N ≈ {n_l3:.0f})")
ax.set_xlabel("Grid size N (interior points per axis)")
ax.set_ylabel("Performance [Mlup/s]")
ax.set_title("Sequential baseline — Poisson, Jacobi")
ax.grid(True)
ax.legend()

# Same curve read in GB/s (Mlup/s * 64 B / 1000) on the right axis
right = ax.secondary_yaxis("right", functions=(lambda m: m * BYTES_PER_LUP / 1e3,
                                               lambda g: g * 1e3 / BYTES_PER_LUP))
right.set_ylabel("Estimated memory bandwidth [GB/s]")

fig.savefig(f"plots/{name}_performance.png", bbox_inches="tight")
plt.close(fig)
