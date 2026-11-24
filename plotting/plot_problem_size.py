"""Choosing N: parallel efficiency and communication share vs number of nodes, one line per N.

Usage (from the repo folder):  python3 plotting/plot_problem_size.py
Reads   results/size_test.csv         (one row per run: jobs/size_test.sub, jobs/size_test_numba.sub)
        results/size_test_ranks.csv   (one row per rank of every run)
Writes  plots/problem_size.svg

Only cube runs with 24 ranks x 1 thread per node are plotted.
Efficiency = T(1 node, same N) / (nodes x T): above 1 = superlinear (cache effect),
below 1 = time lost, mostly to communication. See docs/problem_sizes.md.
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no screen needed, so it works on the cluster
import matplotlib.pyplot as plt

CORES_PER_NODE = 24
CHOSEN = {"mpi": 600, "hybrid": 700}            # multi-node N per implementation
COLOUR = {200: "C0", 400: "C1", 600: "C2", 700: "C3"}


def read_runs():
    """Return a list of runs: dict with impl, N, ranks, threads, time_s, comm_share.

    The rank file has no impl column, so each run is paired with its block of rank rows:
    every run appends its rank rows and its run row right after each other, so the k-th
    run with a given (N, ranks) matches the k-th block with that (N, ranks).
    """
    runs = list(csv.DictReader(open("results/size_test.csv")))
    rank_rows = list(csv.DictReader(open("results/size_test_ranks.csv")))

    blocks = defaultdict(list)                     # (N, ranks) -> [comm share of each block]
    i = 0
    while i < len(rank_rows):
        n, p = int(rank_rows[i]["N"]), int(rank_rows[i]["ranks"])
        block = rank_rows[i:i + p]
        comp = sum(float(r["compute_s"]) for r in block)
        comm = sum(float(r["comm_s"]) for r in block)
        blocks[(n, p)].append(comm / (comp + comm))
        i += p

    used = defaultdict(int)
    out = []
    for r in runs:
        key = (int(r["N"]), int(r["ranks"]))
        share = blocks[key][used[key]]
        used[key] += 1
        out.append(dict(impl=r["impl"], N=key[0], ranks=key[1], threads=int(r["threads"]),
                        time_s=float(r["time_s"]), comm_share=share))
    return out


def per_node(runs, prefix):
    """{N: {nodes: (mean time, mean comm share)}} for cube runs, 1 thread/rank, full nodes."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if not r["impl"].startswith(f"{prefix}-cube") or r["threads"] != 1:
            continue
        if r["ranks"] % CORES_PER_NODE:
            continue
        nodes = r["ranks"] // CORES_PER_NODE
        acc[r["N"]][nodes].append((r["time_s"], r["comm_share"]))
    return {N: {n: tuple(np.mean(v, axis=0)) for n, v in d.items()} for N, d in acc.items()}


runs = read_runs()
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)

for col, (prefix, name) in enumerate([("mpi", "NumPy MPI"), ("hybrid", "Numba")]):
    data = per_node(runs, prefix)
    ax_eff, ax_comm = axes[0, col], axes[1, col]
    for N in sorted(data):
        d = data[N]
        if 1 not in d or len(d) < 2:
            continue                               # no 1-node reference, or nothing to compare it with
        nodes = np.array(sorted(d))
        t = np.array([d[n][0] for n in nodes])
        share = np.array([d[n][1] for n in nodes])
        eff = d[1][0] / (nodes * t)
        chosen = N == CHOSEN[prefix]
        style = dict(color=COLOUR[N], marker="o", linewidth=2.8 if chosen else 1.5,
                     label=f"N = {N}" + ("  (chosen)" if chosen else ""))
        ax_eff.plot(nodes, eff, **style)
        ax_comm.plot(nodes, 100 * share, **style)

    ax_eff.axhline(1.0, color="k", linestyle="--", label="ideal")
    ax_eff.set_title(name)
    ax_eff.set_ylabel("Parallel efficiency  T(1 node) / (n · T)")
    ax_eff.legend(fontsize="small")
    ax_comm.set_ylabel("Time in communication [%]")
    ax_comm.set_xlabel("Nodes (24 ranks per node)")
    ax_comm.set_ylim(bottom=0)
    for ax in (ax_eff, ax_comm):
        ax.grid(True, alpha=0.4)

all_nodes = sorted({r["ranks"] // CORES_PER_NODE for r in runs
                    if r["threads"] == 1 and r["ranks"] % CORES_PER_NODE == 0})
for ax in axes[1]:
    ax.set_xticks(all_nodes)

fig.suptitle("Choosing N: parallel efficiency and communication vs nodes\n"
             f"N = 200–700 · {all_nodes[0]}–{all_nodes[-1]} nodes · 24 ranks/node · "
             "1 thread/rank · cube")
fig.tight_layout()
os.makedirs("plots", exist_ok=True)
fig.savefig("plots/problem_size.svg", bbox_inches="tight")
plt.close(fig)
print("wrote plots/problem_size.svg")
