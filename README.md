# Poisson Problem using MPI

## Who is working on what

| Who | Task |
|-----|------|
| Rasmus | Hybrid MPI implementation with Numba |
| Jesper | Testing geometries for the data |
| Oriol | Investigating why the plot converges at 0.45 and not lower |
| Frederic | Experiments and plotting |

## Getting started

Load the course modules (on the DTU HPC):

```shell
module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8
```

Run from the repo folder, with `python3` (not `python`):

```shell
python3 main_sequential.py -N 30 # run until converged
python3 main_sequential.py -N 200 --max-iter 100 --tol 0 --csv results/baseline.csv   # timing run
mpirun -n 4 python3 main_mpi.py -N 30 # MPI run on 4 ranks
python3 main_sequential.py -h # list all options
```

Check: `-N 30` gives 680 iterations and a max error of 3.42e-03, for the sequential
solver and for MPI on any number of ranks.

### Command-line arguments

| Argument | Default | Meaning |
|----------|---------|---------|
| `-N` | `10` | Number of **interior** grid points per axis; arrays are `(N+2)³` (see [docs/conventions.md](docs/conventions.md)) |
| `--max-iter` | `5000` | Maximum number of Jacobi iterations |
| `--tol` | `1e-6` | Stop when `‖u_new − u‖₂ < tol`. `--tol 0` disables the check and runs exactly `--max-iter` iterations: use this for all timing and scaling runs |
| `--check-every` | `1` | Compute the convergence norm only every k iterations (the norm costs an extra pass over memory) |
| `--csv FILE` | none | Append one result row to `FILE` (header written if the file is new) |
| `--rank-csv FILE` | none | MPI only: append one row per rank with its compute and communication time |
| `--slice FILE` | none | Sequential only: save the middle z-slice of `u` as a `.npy` file for plotting |

## Batch jobs

Submit from the repo folder; output files go to `jobs/output/`, results to `results/`:

```shell
bsub < jobs/baseline.sub     # sequential, fixed iterations over a range of N
bsub < jobs/mpi_strong.sub   # MPI strong scaling on one node, 1-24 ranks
bjobs                        # job status
```

## Plots

Each script reads a CSV from `results/` and writes PNG files to `plots/`:

```shell
python3 plotting/plot_baseline.py results/baseline.csv               # Mlup/s vs N, with cache sizes
python3 plotting/plot_scaling.py results/mpi_strong.csv              # time, speedup, efficiency vs ranks
python3 plotting/plot_load_balance.py results/mpi_strong_ranks.csv   # compute/communication time per rank
```

## Files

| Path | Purpose |
|------|---------|
| `main_sequential.py` | Entry point of the sequential solver: parameters → allocate → initialise → timed Jacobi loop → report |
| `main_mpi.py` | Entry point of the MPI solver: 1D slab decomposition, halo exchange, per-rank timing |
| `poisson/` | Shared building blocks, imported by every entry point |
| `poisson/args.py` | Command-line arguments |
| `poisson/grid.py` | Grid spacing, coordinates, allocation of `u`, `u_new`, `f` for any padded block |
| `poisson/problem.py` | Source term, exact solution, max error, Dirichlet boundary conditions |
| `poisson/jacobi.py` | Jacobi kernels (`jacobi_step`, `diff_sq`) and the shared iteration loop `jacobi()` |
| `poisson/decomposition.py` | MPI: slab split over ranks, 1D Cartesian communicator, halo exchange |
| `poisson/metrics.py` | Conversion of a run time into Mlup/s and GB/s |
| `poisson/output.py` | Printed result line, CSV row per run, CSV row per rank, slice dump |
| `plotting/` | Plot scripts (baseline, strong scaling, load balance) |
| `plots/` | Generated plots |
| `docs/conventions.md` | Shared definitions: `N`, grid spacing, index order and memory layout |
| `docs/report_notes.md` | Notes collected for the report |
| `jobs/` | LSF batch scripts for the `hpcintro` queue |
| `jobs/output/` | Output (`.out`) and error (`.err`) files of the batch jobs, incl. CPU and cache info |
| `results/` | CSV files produced by the runs (one row per run, same columns for every implementation) |
| `jacobi_poisson.py` | Oriol's original single-file solver (uses `N` = all points, i.e. `N + 2` in our convention) |

## Adding a new implementation

Reuse the `poisson/` modules and plug your own kernels into the shared loop:

```python
u, iterations, diff = jacobi(u, u_new, f, delta, max_iter, tol,
                             step=my_step, diff_sq=my_diff_sq)
```

- `main_mpi.py` is the example to follow: `mpi_step` does the halo exchange, then
  `jacobi_step` on the local block; `mpi_diff_sq` sums the local `diff_sq` with `allreduce`.
- Threads (Numba): only the kernels change, so they can be selected in an
  existing entry point instead of a new one.
- Report through `poisson.output.report()` so results land in the common CSV format.
