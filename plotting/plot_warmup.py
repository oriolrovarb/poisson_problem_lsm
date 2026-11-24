"""Warm-up: Mlup/s vs number of iterations, one line per N.

Usage (from the repo folder):  python3 plotting/plot_warmup.py
Reads   results/warmup.csv   (jobs/warmup.sub, hybrid Numba, 24 ranks x 1 thread, slab)
Writes  plots/warmup.svg

Shows how many iterations a run needs before Mlup/s stops rising ("top speed").
Points are the mean of the repetitions, error bars the min and max.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no screen needed, so it works on the cluster
import matplotlib.pyplot as plt

d = np.atleast_1d(np.genfromtxt("results/warmup.csv", delimiter=",", names=True,
                                dtype=None, encoding=None))
ranks, threads = int(d["ranks"][0]), int(d["threads"][0])

fig, ax = plt.subplots(figsize=(7, 4.5))
for c, N in enumerate(np.unique(d["N"])):
    rows = d[d["N"] == N]
    iters = np.unique(rows["iterations"])
    m = [rows["mlups"][rows["iterations"] == i] for i in iters]
    mean = np.array([x.mean() for x in m])
    lo = mean - np.array([x.min() for x in m])
    hi = np.array([x.max() for x in m]) - mean
    gb = 3 * (N + 2) ** 3 * 8 / 1e9
    ax.errorbar(iters, mean, yerr=[lo, hi], marker="o", capsize=3, color=f"C{c}",
                label=f"N = {N} ({gb:.1f} GB, {gb / ranks * 1e3:.0f} MB per rank)")

ax.set_xscale("log")
ax.set_xticks(iters)
ax.set_xticklabels(iters)
ax.set_ylim(bottom=0)
ax.set_xlabel("Jacobi iterations per run")
ax.set_ylabel("Performance [Mlup/s]")
ax.set_title("Warm-up: performance vs iterations, hybrid MPI + Numba\n"
             f"1 node · {ranks} ranks · {threads} thread/rank · slab {ranks}×1×1")
ax.grid(True, which="both", alpha=0.4)
ax.legend(fontsize="small")

os.makedirs("plots", exist_ok=True)
fig.savefig("plots/warmup.svg", bbox_inches="tight")
plt.close(fig)
print("wrote plots/warmup.svg")
