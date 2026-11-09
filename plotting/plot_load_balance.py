"""Load balance: compute, communication and idle time per rank, one plot per (N, ranks).

Usage (from the repo folder):  python3 plotting/plot_load_balance.py results/mpi_ranks.csv
Plots are written to plots/<csv name>_load_balance_N<N>_p<p>.png
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

# Columns: N, ranks, rank, compute_s, comm_s (see poisson/output.py)
data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
nprocs = data["ranks"].astype(int)
ranks = data["rank"].astype(int)
compute = data["compute_s"]
comm = data["comm_s"]
sizes = data["N"].astype(int)

for N, p in sorted(set(zip(sizes, nprocs))):
    if p == 1:
        continue                                  # nothing to balance
    sel = (sizes == N) & (nprocs == p)
    # Average the repetitions for each rank
    r_ids = np.arange(p)
    t_comp = np.array([compute[sel & (ranks == r)].mean() for r in r_ids])
    t_comm = np.array([comm[sel & (ranks == r)].mean() for r in r_ids])
    # No separate "idle" bar: ranks synchronise in every halo exchange, so a
    # fast rank waits INSIDE the communication and all ranks end together.
    # Imbalance shows up as more communication time on the faster ranks.

    plt.figure()
    plt.bar(r_ids, t_comp, label="Computing")
    plt.bar(r_ids, t_comm, bottom=t_comp, label="Communication (incl. waiting)")
    plt.xlabel("Rank")
    plt.ylabel("Time [s]")
    comm_frac = t_comm.sum() / (t_comp + t_comm).sum()
    plt.title(f"Load balance, {p} processes, N = {N} — {comm_frac:.0%} of time in communication")
    plt.xticks(r_ids)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    plt.savefig(f"plots/{name}_load_balance_N{N}_p{p}.png", bbox_inches="tight")
    plt.close()
