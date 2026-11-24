"""Warm-up summary: Mlup/s vs problem size N, for 1, 5 and 50 iterations per run.

Usage (from the repo folder):  python3 plotting/plot_warmup_vs_n.py
Reads   results/warmup.csv   (jobs/warmup.sub, hybrid Numba, 24 ranks x 1 thread, slab)
Writes  plots/warmup_vs_n.svg

The gap between the lines is the warm-up cost: it is large at N = 200 and
almost gone from N = 800, where 1-5 iterations already reach top speed.
Points are the mean of the repetitions, error bars the min and max.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no screen needed, so it works on the cluster
import matplotlib.pyplot as plt

ITERS = [(1, "C0"), (5, "C1"), (50, "C2")]

d = np.atleast_1d(np.genfromtxt("results/warmup.csv", delimiter=",", names=True,
                                dtype=None, encoding=None))
ranks, threads = int(d["ranks"][0]), int(d["threads"][0])
Ns = np.unique(d["N"])

fig, ax = plt.subplots(figsize=(7, 4.5))
for it, colour in ITERS:
    m = [d["mlups"][(d["N"] == N) & (d["iterations"] == it)] for N in Ns]
    mean = np.array([x.mean() for x in m])
    lo = mean - np.array([x.min() for x in m])
    hi = np.array([x.max() for x in m]) - mean
    ax.errorbar(Ns, mean, yerr=[lo, hi], marker="o", capsize=3, color=colour,
                label=f"{it} iteration{'s' if it > 1 else ''} per run")

# x tick labels: N and the data per rank (3 arrays of (N+2)^3 doubles)
ax.set_xticks(Ns)
ax.set_xticklabels([f"{N}\n{3 * (N + 2) ** 3 * 8 / ranks / 1e6:.0f} MB/rank" for N in Ns])
ax.set_ylim(bottom=0)
ax.set_xlabel("Problem size N (data per rank)")
ax.set_ylabel("Performance [Mlup/s]")
ax.set_title("Warm-up vs problem size, hybrid MPI + Numba\n"
             f"1 node · {ranks} ranks · {threads} thread/rank · slab {ranks}×1×1")
ax.grid(True, alpha=0.4)
ax.legend(loc="lower right")

os.makedirs("plots", exist_ok=True)
fig.savefig("plots/warmup_vs_n.svg", bbox_inches="tight")
plt.close(fig)
print("wrote plots/warmup_vs_n.svg")
