"""Result output: one readable line per run, optionally one CSV row, slice dump.

Every implementation reports through report() with the same fields, so all
experiments land in the same CSV format and can be plotted by one script.
"""
import csv
import os

import numpy as np

CSV_FIELDS = [
    "impl",         # e.g. "numpy-seq", "mpi-slab", "hybrid-numba"
    "N",            # interior points per axis
    "ranks",        # number of MPI ranks (1 for sequential)
    "threads",      # threads per rank (1 unless threaded)
    "iterations",   # iterations actually done
    "max_iter",
    "tol",
    "check_every",
    "time_s",       # wall-clock time of the iteration loop only
    "mlups",
    "gbs",
    "final_diff",   # NaN for timing runs (tol <= 0)
    "max_error",    # max |u - u_exact| over the interior
]


def report(result, csv_path=None):
    """Print one readable line; append result as a CSV row if csv_path is given.

    result is a dict with (at least) the keys in CSV_FIELDS. A header row is
    written only when the file does not exist yet, so many runs can append
    to the same file.
    """
    print(f"[{result['impl']}] N={result['N']} ranks={result['ranks']} "
          f"threads={result['threads']} | iterations={result['iterations']} "
          f"| time={result['time_s']:.3f} s | {result['mlups']:.2f} Mlup/s "
          f"({result['gbs']:.2f} GB/s) | final diff={result['final_diff']:.2e} "
          f"| max error={result['max_error']:.2e}")
    if csv_path:
        new_file = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(result)


def save_slice(u, path, k=None):
    """Save the 2D slice u[:, :, k] (default: middle) as .npy for plotting.

    Placeholder until we get the slice-dump method the assignment mentions.
    """
    k = u.shape[2] // 2 if k is None else k
    np.save(path, u[:, :, k])