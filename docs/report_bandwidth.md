# Report section: memory bandwidth and the limiting factor

Draft for the report. Numbers from `results/stream.csv` (job `jobs/stream.sub`, 1 run) and the solver
runs `results/mpi_decomp.csv` / `results/mpi_placement_socket.csv` (N = 200, 100 iterations, see
`docs/report_decomposition.md`). Intel Xeon E5-2650 v4, 2 sockets × 12 cores, `hpcintro` queue.
Figure: `plots/bandwidth.png` (`python3 plotting/plot_bandwidth.py`).

## 1. Hardware peak

Each socket has 4 DDR4-2400 memory channels of 8 bytes:
4 × 2400·10⁶ × 8 B = **76.8 GB/s per socket, 153.6 GB/s per node** (matches Intel's specification).
This is the maximum supported memory speed; the speed of the installed DIMMs could not be checked
(no root access).

## 2. Achievable bandwidth (STREAM)

STREAM 5.10, compiled with `gcc -O3 -march=native -fopenmp`, 40·10⁶ doubles (320 MB) per array,
more than 4× the 60 MB of L3 per node. Threads pinned one per core (`OMP_PLACES=cores`), either
filling socket 0 first (`close`) or alternating sockets (`spread`); the whole node was reserved.

| threads | 1 | 2 | 4 | 8 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|
| Triad, close [GB/s] | 10.7 | 19.9 | 34.5 | 45.2 | 46.5 | 60.5 | 76.9 | 92.5 |
| Triad, spread [GB/s] | 10.7 | 21.3 | 39.5 | 68.4 | 83.5 | 73.4 | 77.3 | 91.4 |

- One core reaches 10.7 GB/s; one socket saturates at about 46 GB/s after 8 cores (`close`, 8–12
  threads); the full node reaches about 92 GB/s, i.e. **60 % of the theoretical peak**, for both
  sockets alike (46 / 76.8 = 60 %).
- Spreading threads over both sockets uses both memory controllers early: +50 % at 8 threads,
  +80 % at 12.
- The dip for `spread` at 16 threads (73 GB/s, lower than at 12) is from a single run and was not
  investigated; it should be repeated before drawing conclusions from it.
- STREAM counts 16/24 B per element and ignores the write-allocate read of the target array, so
  the actual bus traffic is up to 1.33–1.5× higher; STREAM is a conservative, achievable reference.

## 3. Our solver against STREAM

With the course convention of 64 B per lattice update, the solver reaches 3.0 GB/s on one core,
16 GB/s at 12 ranks (core mapping) and 33 GB/s at 24 ranks, i.e. only about **36 % of STREAM** at
24 ranks. Taken at face value this would suggest that the solver is not limited by memory bandwidth.

However, the ratio STREAM / solver is almost constant: 3.6 on one core and 2.8–2.9 from 8 to 24
ranks, for both placements. The solver curves also have the same shape as STREAM: the plateau
between 8 and 12 ranks with core mapping, and the gain from socket mapping, appear in both
(`plots/bandwidth.png`). A constant factor with the same shape is what one expects if the solver
is memory-bound and moves about 3× more bytes per update than the 64 B convention assumes.

The NumPy implementation explains this factor. The update
`u_new[1:-1,1:-1,1:-1] = (1/6)*(u[:-2,..] + u[2:,..] + ... + delta2*f[..])` is evaluated as
about 8 separate array operations (6 additions, `delta2*f`, `*1/6`) plus the copy into `u_new`.
Each operation streams one or two full arrays from memory and writes a temporary array, about
16–24 B per point each, i.e. roughly 150–200 B per update instead of 64 B. This is an estimate
from counting operations, not a measurement (no hardware counters were used).

## 4. Conclusion: the limiting factor

**The solver is limited by memory bandwidth.** It follows the STREAM curve at a constant factor
of about 3, which matches the extra memory traffic of the NumPy temporaries. Consequences:

- Adding ranks only helps while memory bandwidth is still free; that is why the speedup flattens
  between 8 and 12 ranks on one socket, and why rank placement matters more than the decomposition
  geometry (`docs/report_decomposition.md`).
- The way to go faster on one node is to move fewer bytes per update, not to add more ranks: a
  compiled kernel (e.g. Numba) that does the whole update in one pass without temporaries could
  get close to the 64 B (or less, with cache reuse of neighbours) per update.

## 5. Limitations

- STREAM was run once; the solver was not run with 20 ranks and socket mapping above 16 ranks.
- The 3× traffic factor is an estimate from counting operations; it is consistent with the
  measurements but not measured directly.
- STREAM uses OpenMP threads and the solver uses MPI processes. Both keep their data in memory on
  their own socket (first touch), so they use the memory controllers in the same way.
