# Problem size N per experiment

All timing runs: `--max-iter 100 --tol 0`, except one-node hybrid (5 iterations, see table). Hardware (hpcintro): 6 nodes, each 2 × Xeon E5-2650 v4
(24 cores, L3 = 2 × 30 MB = **60 MB per node**), jobs use `-M 4GB` per rank.
Data size = arrays `u, u_new, f` = `3·(N+2)³·8 B`.

| Scenario | N | Data | Why |
|---|---|---|---|
| Sequential baseline | sweep 10 … 400 | 0.03 MB … 1.56 GB | the curve shows the cache drop at N ≈ 105 (data = 30 MB = L3 of one socket). Largest run: 128 s (under the assignment's 3–4 min limit), measured peak memory 3.06 GB (< 4 GB) |
| One node, NumPy MPI, 1–24 ranks | **200** | 198 MB = 3.3 × node L3 | out of cache, 24-rank run takes 1.5 s, existing results stay valid |
| One node, Numba / hybrid ([jobs/hybrid_n800.sub](../jobs/hybrid_n800.sub)) | **800**, `--max-iter 5`, `-M 16GB` | 12.4 GB = 206 × node L3 | chosen by Rasmus from the warm-up study ([plots/warmup.svg](../plots/warmup.svg), 24 ranks × 1 thread): at N=800, 5 iterations give 2362 Mlup/s, within 1 % of 50 iterations (2365), so start-up cost is gone. A run takes 1.1 s. The 1-rank run holds all 12.4 GB, hence `-M 16GB`. The NumPy reference in that job also runs at N=800, 5 iterations |
| Multi node, NumPy MPI, 2–6 nodes | **600** | 5.2 GB = 2.6 GB/node (2 nodes) … 870 MB/node (6 nodes) = 44 … 15 × L3 | at 3 nodes (measured): efficiency 97 % and communication 11 %. N=400 already drops to 84 % and 31 %. At 6 nodes N=600 should still give ≈ 70 ms compute per iteration and ≈ 7 s per run. Speedup reference: 1 full node (41.8 s); a 1-core run does not fit in 4 GB |
| Multi node, Numba / hybrid, 2–6 nodes | **700** | 8.3 GB = 4.1 GB/node (2 nodes) … 1.4 GB/node (6 nodes) | Numba computes ~6× faster, so it needs more points per node. At N=400 on 3 nodes the run takes 0.84 s and communication is 30 % (measured). N=700 on 3 nodes: 4.1 s, 35 ms compute per iteration, 19 % communication, so ≈ 2 s and ≈ 18 ms/it at 6 nodes |

Figure: [plots/problem_size.svg](../plots/problem_size.svg) (`python3 plotting/plot_problem_size.py`),
efficiency and communication share vs nodes for every tested N.

**Decomposition on several nodes: use cube (`--dims` omitted) or pencil, not slab.** This is not about N.
A slab cuts only one axis, so each rank gets N/P planes. With 144 ranks and N=600 that is 4.2 planes,
and the 2 halo planes each rank exchanges are 48 % of what it computes (cube: 5 %). Measured at 3 nodes,
N=400: slab spends 51 % of its time in communication, cube 31 %.

## Measurements ([jobs/size_test.sub](../jobs/size_test.sub), NumPy, 24 ranks per node)

`compute` = slowest rank's compute time per iteration, `comm` = mean share of time in halo exchange + allreduce.

| Nodes | N | Data per node | Time | Mlup/s | compute / it | comm (cube / slab) |
|---|---|---|---|---|---|---|
| 1 | 200 | 198 MB (3.3 × L3) | 1.75 s | 456 | 15.5 ms | 45 % / 35 % |
| 1 | 400 | 1.56 GB (26 × L3) | 11.8 s | 542 | 115 ms | 7 % / 17 % |
| 1 | 600 | 5.2 GB (87 × L3) | 41.8 s | 516 | 411 ms | 2 % / 6 % |
| 2 | 200 | 99 MB (1.6 × L3) | 0.56 s | 1427 | 4.2 ms | 31 % / 43 % |
| 2 | 400 | 780 MB (13 × L3) | 6.4 s | 1004 | 60 ms | 10 % / 15 % |
| 2 | 600 | 2.6 GB (44 × L3) | 22.5 s | 962 | 216 ms | 8 % / 10 % |
| 3 | 200 | 66 MB (1.1 × L3) | 0.49 s | 1621 | 2.1 ms | 59 % / 47 % |
| 3 | 400 | 520 MB (8.7 × L3) | 4.7 s | 1367 | 40 ms | 31 % / 51 % |
| 3 | 600 | 1.7 GB (29 × L3) | 14.4 s | 1499 | 137 ms | 11 % / 20 % |
| 4, 6 | | | *jobs pending* | | | |

Efficiency T(1 node) / (nodes × T), cube: N=400 93 % (2 nodes), 84 % (3 nodes); N=600 93 %, 97 %.

What this shows:

- **N=200 is wrong for multi node.** Going from 1 to 2 nodes, the time falls 3.1× and the compute per iteration
  falls 3.7×, for only 2× the cores. That is superlinear: the 99 MB per node almost fits in L3.
  From 4 nodes on, the whole problem (≤ 49 MB per node) fits in cache.
- **N=400 is fine on 2 nodes but not beyond.** It scales well to 2 nodes (1.85×, compute 115 → 60 ms).
  At 3 nodes the communication share is 31 %. Most of that is waiting: compute per rank ranges from 26 to
  40 ms, and the fastest rank's exchange takes 6.4 ms. At 6 nodes it would get worse.
- **N=600 keeps every rank busy up to 6 nodes:** 137 ms of compute per iteration at 3 nodes and 11 %
  communication, so ≈ 70 ms at 6 nodes.

### Numba, one node ([jobs/size_test_numba.sub](../jobs/size_test_numba.sub))

These runs led to N=400 first. The team then chose N=800 with 5 iterations (table above); they are kept as data.

| N | ranks × threads | Time | Mlup/s | compute / it | comm |
|---|---|---|---|---|---|
| 200 | 1 × 1 | 2.43 s | 330 | 23.5 ms | 3 % |
| 200 | 24 × 1 (cube) | 0.28 s | 2907 | 2.3 ms | 21 % |
| 200 | 6 × 4 | 0.30 s | 2666 | 2.5 ms | 20 % |
| 400 | 1 × 1 | 19.1 s | 336 | 186 ms | 2 % |
| 400 | 24 × 1 (cube) | 1.98 s | 3237 | 17.7 ms | 12 % |
| 400 | 6 × 4 | 2.77 s | 2314 | 24.3 ms | 14 % |

- **Numba needs a larger N than NumPy.** It is ~6× faster per core, so at N=200 a full node finishes in 0.3 s
  and communication takes 20 % of the time. At N=400 the run takes 2–3 s and communication drops to 12–14 %.

### Numba, several nodes (24 ranks × 1 thread per node, cube)

Efficiency = T(1 node, same N) / (nodes × T). 1-node values: mean of 2 runs at N=400, 1 run at N=600 and 700.

| Nodes | N | Data per node | Time | Mlup/s | Efficiency | compute / it | comm |
|---|---|---|---|---|---|---|---|
| 1 | 400 | 1.56 GB | 1.99 s | 3220 | 100 % | 17.9 ms | 12 % |
| 2 | 400 | 780 MB | 1.14 s | 5615 | 87 % | 9.4 ms | 22 % |
| 3 | 400 | 520 MB | 0.84 s | 7600 | 79 % | 6.2 ms | 30 % |
| 1 | 600 | 5.2 GB | 8.16 s | 2646 | 100 % | 76.1 ms | 15 % |
| 2 | 600 | 2.6 GB | 3.68 s | 5873 | 111 % | 31.9 ms | 18 % |
| 3 | 600 | 1.7 GB | 2.57 s | 8390 | 106 % | 21.6 ms | 22 % |
| 1 | 700 | 8.3 GB | 13.68 s | 2507 | 100 % | 128.5 ms | 9 % |
| 2 | 700 | 4.1 GB | 6.32 s | 5424 | 108 % | 54.8 ms | 19 % |
| 3 | 700 | 2.8 GB | 4.09 s | 8382 | 111 % | 35.5 ms | 19 % |

- **N=400 is too small for Numba on several nodes.** Communication grows 12 → 22 → 30 % and efficiency drops
  to 79 % already at 3 nodes. At 6 nodes the run would last ≈ 0.4 s.
- **N=600 and N=700 scale slightly superlinearly (106–111 %).** This is a smaller cache effect than with
  N=200, because the full data never fits (≥ 1.7 GB per node). Each Jacobi sweep reuses 3 neighbouring planes
  of `u` per rank. On one node at N=700 these take 1.96 MB per rank, which overflows a core's ~2.5 MB share of
  L3 once `f` and `u_new` stream through too. On 2–3 nodes the planes shrink to 0.98 MB and stay in cache.
  The same effect explains why one node gets fewer Mlup/s at large N (3220 → 2646 → 2507 for N = 400 → 600 → 700).
  This is the most likely explanation, not a proven one. It has to be explained in the report, but no N avoids
  it: a smaller N removes it only by making communication dominate.
- **Choose N=700:** at 6 nodes the run still lasts ≈ 2 s with ≈ 18 ms compute per iteration
  (N=600: ≈ 1.3 s, ≈ 11 ms).
- 6 ranks × 4 threads per node is slower than 24 × 1 at N ≥ 600 (e.g. 3 nodes, N=700: 5.30 s vs 4.09 s).
- The speedup reference for Numba multi-node is **one full node at N=700** (13.7 s). A 1-core run is not
  possible there: 8.3 GB of data does not fit in `-M 4GB`.
