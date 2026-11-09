# Conventions

This contains the shared definitions for every implementation in this project (sequential, MPI, hybrid).

## Grid size `N`

- `N` is the number of interior grid points per axis (the unknowns updated by Jacobi).
- Every 3D array (`u`, `u_new`, `f`) has shape `(N+2, N+2, N+2)`: one extra boundary layer on each side.
- Along each axis, index `0` and index `N+1` are boundary points and indices `1 … N` are **interior** points.

## Grid spacing `Δ`

- Along each axis there are `N+2` points spanning `[-1, 1]`, so `N+1` gaps of equal size:
  `Δ = 2 / (N + 1)`.
- Point `i` sits at `x_i = -1 + i·Δ` (so `x_0 = -1` and `x_{N+1} = 1`); the same holds for `y_j` and `z_k`.

## Index order and memory layout

- Arrays are indexed `u[i, j, k] = u(x_i, y_j, z_k)`.
- NumPy stores arrays row-major (C order): the last index `k` is contiguous in memory,
  `i` is the slowest (neighbours in `i` are a whole plane, `(N+2)²` values, apart).
- Loops over the grid must have `k` innermost.
