"""Strong scaling plots: time, speedup and parallel efficiency vs number of ranks.

Usage (from the repo folder):  python3 plotting/plot_scaling.py results/mpi_strong.csv
Plots are written to plots/<csv name>_*.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no screen needed, so it works on the cluster
import matplotlib.pyplot as plt

csv_path = sys.argv[1]
name = os.path.splitext(os.path.basename(csv_path))[0]
os.makedirs("plots", exist_ok=True)

# Read the columns we need by name (the CSV has a header row, see poisson/output.py)
data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
ranks = np.atleast_1d(data["ranks"]).astype(int)
time_s = np.atleast_1d(data["time_s"])
N = np.atleast_1d(data["N"])[0]

# Average the repetitions for each number of ranks
p_values = np.unique(ranks)
t_mean = np.array([time_s[ranks == p].mean() for p in p_values])

# Reference run: normally p = 1. If 1 is missing, use the smallest p
# (relative speedup, scaled so that it equals p0 at p0).
p0 = p_values[0]
t0 = t_mean[0]
speedup = p0 * t0 / t_mean
efficiency = speedup / p_values

title_suffix = f"Poisson, slab decomposition, N = {N}"
plt.rcParams["axes.titlepad"] = 30     # room for the top axis


def add_work_per_rank_axis():
    """Top axis: how many grid points each rank computes (N^3 / p), in millions.

    In strong scaling the total work is fixed, so each rank's share shrinks.
    """
    top = plt.gca().secondary_xaxis("top")
    top.set_xticks(p_values)
    top.set_xticklabels([f"{float(N)**3 / p / 1e6:.2g}" for p in p_values])
    top.set_xlabel("Grid points per rank [millions]")


# 1. Total time
plt.figure()
plt.plot(p_values, t_mean, "o-")
plt.xlabel("Number of MPI processes")
plt.ylabel("Time of the iteration loop [s]")
plt.title(f"Time — {title_suffix}")
plt.xticks(p_values)
add_work_per_rank_axis()
plt.grid(True)
plt.savefig(f"plots/{name}_timing.png", bbox_inches="tight")
plt.close()

# 2. Speedup with ideal line
plt.figure()
plt.plot(p_values, speedup, "o-", label="Measured")
plt.plot(p_values, p_values, "k--", label="Ideal (S = p)")
plt.xlabel("Number of MPI processes")
plt.ylabel("Speedup")
plt.title(f"Speedup — {title_suffix}")
plt.xticks(p_values)
add_work_per_rank_axis()
plt.legend()
plt.grid(True)
plt.savefig(f"plots/{name}_speedup.png", bbox_inches="tight")
plt.close()

# 3. Parallel efficiency with ideal line at 1
plt.figure()
plt.plot(p_values, efficiency, "o-", label="Measured")
plt.axhline(1.0, color="k", linestyle="--", label="Ideal (E = 1)")
plt.xlabel("Number of MPI processes")
plt.ylabel("Parallel efficiency  S(p) / p")
plt.title(f"Parallel efficiency — {title_suffix}")
plt.xticks(p_values)
add_work_per_rank_axis()
plt.ylim(0, max(1.1, 1.05 * efficiency.max()))   # superlinear (E > 1) stays visible
plt.legend()
plt.grid(True)
plt.savefig(f"plots/{name}_efficiency.png", bbox_inches="tight")
plt.close()
