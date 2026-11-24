# Plotting rules

Every plot in the report follows these rules, so readers can compare figures from different people.

## 1. File format: always `.svg`

```python
fig.savefig("plots/<name>.svg", bbox_inches="tight")
```

Use no `.png` and no `.pdf`. SVG stays sharp at any zoom in the report.

## 2. Every plot states the run setup

A reader must be able to see these from the plot alone (title, axis labels or legend):

| What | How to write it | When |
|---|---|---|
| Problem size | `N = 400` | always |
| Nodes | `1 node`, `3 nodes` | always, also when it is 1 |
| MPI ranks | `24 ranks` | always |
| Threads per rank | `4 threads/rank` | only for Numba runs. Then cores = ranks × threads, which differs from the rank count |
| Decomposition | `slab 24×1×1`, `pencil 6×4×1`, `cube 4×3×2` | every MPI run |

If a value changes across the plot (for example ranks on the x-axis), put it on the axis and leave it out of the title.
The title only holds what is fixed for the whole figure.

## 3. Title layout: what on line 1, setup on line 2

Two lines keep a long title readable:

```python
ax.set_title("Strong scaling, NumPy MPI\n"
             "N = 600 · 1–6 nodes · 24 ranks/node · cube")
```

```python
ax.set_title("Hybrid MPI + Numba, ranks × threads\n"
             "N = 400 · 1 node · 24 cores · slab")
```

- Line 1: what is measured (experiment and implementation).
- Line 2: the setup, separated by ` · `.
- For a figure with several subplots, put the common setup in `fig.suptitle` and keep each subplot title short.

## 4. Colours: blue first, then orange

Use matplotlib's default colours in this order, as in `plots/baseline_performance`:

| Order | Colour | Code | Use for |
|---|---|---|---|
| 1 | blue | `"C0"` | the main measured data |
| 2 | orange | `"C1"` | the second series or the comparison |
| reference | black | `"k"`, dashed | ideal lines, hardware limits, cache sizes |

Set the colours explicitly (`color="C0"`, `color="C1"`), so they don't depend on plotting order.
If you need more than two data series, continue with `"C2"` (green) and `"C3"` (red).

## Checklist before committing a plot

- [ ] Saved as `.svg` in `plots/`
- [ ] N, nodes, ranks and decomposition visible; threads/rank if Numba
- [ ] Two-line title: what / setup
- [ ] Blue = main data, orange = comparison, black dashed = reference
- [ ] Axis labels with units (`time [s]`, `performance [Mlup/s]`)
