"""Hybrid MPI + Numba: thread scaling and ranks x threads mixes on one node.

Usage (from the repo folder):  python3 plotting/plot_hybrid.py
Reads   results/hybrid.csv       (jobs/hybrid.sub, Numba kernels, slab decomposition)
        results/mpi_decomp.csv   (NumPy kernels, mpi-slab-24x1x1, reference)
Writes  plots/hybrid.png

Left:  strong scaling speedup T(1)/T(t) of 1 rank with t Numba threads, log-log,
       with the ideal line S = t.
Right: Mlup/s of every ranks x threads mix that uses all 24 cores, against the
       NumPy MPI solver on 24 ranks.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no screen needed, so it works on the cluster
import matplotlib.pyplot as plt

BYTES_PER_LUP = 64             # course convention, as in poisson/metrics.py
CORES = 24


def read(path):
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True,
                                       dtype=None, encoding=None))


def mean_per_config(d):
    """Mean time and Mlup/s for each (ranks, threads) pair, sorted by ranks."""
    configs = sorted({(int(r), int(t)) for r, t in zip(d["ranks"], d["threads"])})
    rows = []
    for r, t in configs:
        sel = (d["ranks"] == r) & (d["threads"] == t)
        rows.append((r, t, d["time_s"][sel].mean(), d["mlups"][sel].mean()))
    return rows


hybrid = read("results/hybrid.csv")
N = int(hybrid["N"][0])
rows = mean_per_config(hybrid)

# references
decomp = read("results/mpi_decomp.csv")
numpy_24 = decomp["mlups"][decomp["impl"] == "mpi-slab-24x1x1"].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

# 1. thread scaling: 1 rank, t threads
single = [(t, time_s) for r, t, time_s, _ in rows if r == 1]
threads = np.array([t for t, _ in single])
times = np.array([time_s for _, time_s in single])
speedup = times[threads == 1][0] / times
ax1.loglog(threads, speedup, "o-", label="Measured (1 rank, Numba)")
ax1.loglog(threads, threads, "k--", label="Ideal (S = t)")
ax1.set_xticks(threads)
ax1.set_xticklabels(threads)
ax1.set_yticks(threads)
ax1.set_yticklabels(threads)
ax1.set_xlabel("Numba threads (1 MPI rank)")
ax1.set_ylabel("Speedup  T(1) / T(t)")
ax1.set_title(f"Thread scaling, N = {N}")
ax1.grid(True, which="both", alpha=0.4)
ax1.legend()

# 2. ranks x threads mixes on all 24 cores
mixes = [(r, t, m) for r, t, _, m in rows if r * t == CORES]
mixes.sort(key=lambda x: -x[0])                  # 24x1 first, 1x24 last
labels = [f"{r}×{t}" for r, t, _ in mixes]
ax2.bar(labels, [m for _, _, m in mixes], color="C0", label="Hybrid MPI + Numba")
ax2.axhline(numpy_24, color="C1", linestyle="--",
            label=f"NumPy MPI, 24 ranks ({numpy_24:.0f} Mlup/s)")
ax2.set_xlabel("MPI ranks × Numba threads (24 cores, one node)")
ax2.set_ylabel("Performance [Mlup/s]")
ax2.set_title(f"Ranks × threads on 24 cores, N = {N}")
ax2.grid(True, axis="y", alpha=0.4)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize="small")

fig.suptitle("Hybrid MPI + Numba — Xeon E5-2650 v4, 2 sockets × 12 cores, slab decomposition")
fig.tight_layout()
os.makedirs("plots", exist_ok=True)
fig.savefig("plots/hybrid.png", bbox_inches="tight", dpi=150)
plt.close(fig)
print("wrote plots/hybrid.png")
