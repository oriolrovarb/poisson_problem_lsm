"""Memory bandwidth: STREAM Triad vs our MPI solver, on one node, against the hardware peak.

Usage (from the repo folder):  python3 plotting/plot_bandwidth.py
Reads   results/stream.csv                  (jobs/stream.sub)
        results/mpi_decomp.csv              (solver, --map-by core,   jobs/mpi_decomp.sub)
        results/mpi_placement_socket.csv    (solver, --map-by socket, jobs/mpi_placement.sub)
Writes  plots/bandwidth.svg
Solver GB/s use the course convention of 64 bytes per lattice update (the `gbs` column);
mean over all layouts (slab/pencil/cube) and repetitions at each number of ranks.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no screen needed, so it works on the cluster
import matplotlib.pyplot as plt

# Xeon E5-2650 v4: 4 channels x DDR4-2400 x 8 B per socket (docs/report_notes.md)
PEAK_SOCKET = 76.8             # GB/s
PEAK_NODE = 2 * PEAK_SOCKET


def read(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)


def solver_mean(path):
    """Mean GB/s (64 B convention) for each number of ranks in a solver CSV."""
    d = read(path)
    ranks = np.unique(d["ranks"])
    return ranks, np.array([d["gbs"][d["ranks"] == p].mean() for p in ranks])


stream = read("results/stream.csv")
triad = stream[stream["kernel"] == "Triad"]

fig, ax = plt.subplots()
for bind, colour in [("close", "C0"), ("spread", "C1")]:
    rows = triad[triad["binding"] == bind]
    ax.plot(rows["threads"], rows["best_MBs"] / 1e3, "o-", color=colour,
            label=f"STREAM Triad, {bind}")

for path, label, colour in [("results/mpi_decomp.csv", "solver, map-by core", "C0"),
                            ("results/mpi_placement_socket.csv", "solver, map-by socket", "C1")]:
    ranks, gbs = solver_mean(path)
    ax.plot(ranks, gbs, "s--", color=colour, label=f"{label} (64 B/update)")

ax.axhline(PEAK_NODE, color="k", linestyle="-", label=f"peak node {PEAK_NODE:.1f} GB/s")
ax.axhline(PEAK_SOCKET, color="k", linestyle=":", label=f"peak socket {PEAK_SOCKET:.1f} GB/s")
ax.set_xlim(0, 25)
ax.set_ylim(0, 165)
ax.set_xlabel("Threads (STREAM) / MPI ranks (solver), one node")
ax.set_ylabel("Memory bandwidth [GB/s]")
ax.set_title("Memory bandwidth: STREAM Triad vs NumPy MPI solver\n"
             "solver N = 200 · 1 node · 1–24 ranks · mean of slab/pencil/cube")
ax.grid(True)
ax.legend(loc="upper left", fontsize="small")

os.makedirs("plots", exist_ok=True)
fig.savefig("plots/bandwidth.svg", bbox_inches="tight")
plt.close(fig)
