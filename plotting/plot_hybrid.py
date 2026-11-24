"""Hybrid MPI + Numba: thread scaling and ranks x threads mixes on one node.

Usage (from the repo folder):
    python3 plotting/plot_hybrid.py                            # N=200, jobs/hybrid.sub
    python3 plotting/plot_hybrid.py results/hybrid_n800.csv    # N=800, jobs/hybrid_n800.sub
Reads   the given CSV (default results/hybrid.csv): hybrid-* rows are the Numba runs.
        NumPy reference on 24 ranks: the mpi-* rows of the same CSV if the job made
        them, else results/mpi_nonblocking_improved.csv at the same N.
Writes  plots/<csv name>.svg

Left:  strong scaling speedup T(1)/T(t) of 1 rank with t Numba threads, log-log,
       with the ideal line S = t.
Right: Mlup/s of every ranks x threads mix that uses all 24 cores, against the
       NumPy MPI solver on 24 ranks.
"""
import os
import sys

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


csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/hybrid.csv"
name = os.path.splitext(os.path.basename(csv_path))[0]
data = read(csv_path)
is_hybrid = np.char.startswith(data["impl"].astype(str), "hybrid-")
hybrid = data[is_hybrid]
N = int(hybrid["N"][0])
iters = int(hybrid["iterations"][0])
rows = mean_per_config(hybrid)

# NumPy reference on 24 ranks: from the same job if it has one, else Oriol's non-blocking run
own = data[~is_hybrid & (data["ranks"] == CORES)]
if len(own):
    numpy_24 = own["mlups"].mean()
    numpy_label = f"NumPy MPI, 24 ranks, slab 24×1×1, same job ({numpy_24:.0f} Mlup/s)"
else:
    nonblock = read("results/mpi_nonblocking_improved.csv")
    numpy_24 = nonblock["mlups"][(nonblock["N"] == N) & (nonblock["ranks"] == CORES)].mean()
    numpy_label = f"NumPy MPI non-blocking, 24 ranks, slab 24×1×1 ({numpy_24:.0f} Mlup/s)"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

# 1. thread scaling: 1 rank, t threads
single = [(t, time_s) for r, t, time_s, _ in rows if r == 1]
threads = np.array([t for t, _ in single])
times = np.array([time_s for _, time_s in single])
speedup = times[threads == 1][0] / times
ax1.loglog(threads, speedup, "o-", color="C0", label="Measured (1 rank, Numba)")
ax1.loglog(threads, threads, "k--", label="Ideal (S = t)")
ax1.set_xticks(threads)
ax1.set_xticklabels(threads)
ax1.set_yticks(threads)
ax1.set_yticklabels(threads)
ax1.set_xlabel("Numba threads (1 MPI rank)")
ax1.set_ylabel("Speedup  T(1) / T(t)")
ax1.set_title("Thread scaling: 1 rank × t threads")
ax1.grid(True, which="both", alpha=0.4)
ax1.legend()

# 2. ranks x threads mixes on all 24 cores
mixes = [(r, t, m) for r, t, _, m in rows if r * t == CORES]
mixes.sort(key=lambda x: -x[0])                  # 24x1 first, 1x24 last
labels = [f"{r}×{t}" for r, t, _ in mixes]
ax2.bar(labels, [m for _, _, m in mixes], color="C0", label="Hybrid MPI + Numba")
ax2.axhline(numpy_24, color="C1", linestyle="--", label=numpy_label)
ax2.set_xlabel("MPI ranks × Numba threads (24 cores, one node)")
ax2.set_ylabel("Performance [Mlup/s]")
ax2.set_title("Ranks × threads = 24 cores")
ax2.grid(True, axis="y", alpha=0.4)
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize="small")

fig.suptitle("Hybrid MPI + Numba\n"
             f"N = {N} · {iters} iterations · 1 node · ranks × threads/rank · slab R×1×1")
fig.tight_layout()
os.makedirs("plots", exist_ok=True)
fig.savefig(f"plots/{name}.svg", bbox_inches="tight")
plt.close(fig)
print(f"wrote plots/{name}.svg")
