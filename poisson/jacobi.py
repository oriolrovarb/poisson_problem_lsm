"""Jacobi iterations for the 7-point stencil.

jacobi() is the iteration loop. It does not care HOW a sweep is done: it calls
two pluggable functions, so other implementations only replace those:

  step(u, u_new, f, delta2)  one sweep, writes the interior of u_new from u
  diff_sq(u, u_new)          sum of squared changes over the interior

  MPI:   step    = halo exchange, then jacobi_step on the local block
         diff_sq = local diff_sq, then comm.allreduce(..., op=MPI.SUM)
  Numba: step / diff_sq = prange versions of the same kernels

Both kernels only touch [1:-1, 1:-1, 1:-1], so they work on the global array
and on an MPI rank's local block with ghost layers alike.
"""
import numpy as np


def jacobi_step(u, u_new, f, delta2):
    """One Jacobi sweep (7-point stencil) over the interior.

    Contract:
      * read only from u, write only to u_new[1:-1, 1:-1, 1:-1]
      * never touch the outer layer of u_new (boundary / ghost values)
      * delta2 is delta**2, precomputed once by the loop
      * returns nothing (u_new is modified in place)

    Stencil taken from Oriol's jacobi_poisson.py (only delta**2 -> delta2).
    """
    # 7-point stencil update for inner grid points
    u_new[1:-1, 1:-1, 1:-1] = (1/6.0)*(
        u[:-2, 1:-1, 1:-1]
        + u[2:, 1:-1, 1:-1]
        + u[1:-1, :-2, 1:-1]
        + u[1:-1, 2:, 1:-1]
        + u[1:-1, 1:-1, :-2]
        + u[1:-1, 1:-1, 2:]
        + delta2 * f[1:-1, 1:-1, 1:-1])


def diff_sq(u, u_new):
    """Sum of squared changes over the interior: sum((u_new - u)**2).

    Contract:
      * interior [1:-1, 1:-1, 1:-1] only, so MPI ghost layers are never
        counted twice
      * return the SQUARED sum as a Python float, not the norm: MPI adds the
        ranks' parts first, the loop takes the sqrt afterwards

    Oriol's Frobenius norm of the change, restricted to the interior and squared.
    """
    return float(np.linalg.norm(u_new[1:-1, 1:-1, 1:-1] - u[1:-1, 1:-1, 1:-1])**2)


def jacobi(u, u_new, f, delta, max_iter, tol, check_every=1,
           step=jacobi_step, diff_sq=diff_sq):
    """Run Jacobi iterations; return (u, iterations, final_diff).

    u and u_new must both contain the boundary values on entry: the arrays are
    swapped after every sweep instead of copied, so each one is "the new array"
    every other iteration. tol <= 0 disables the convergence check (fixed
    number of iterations, for timing); final_diff is then NaN.
    """
    delta2 = delta * delta
    diff = float("nan")
    it = 0
    for it in range(1, max_iter + 1):
        step(u, u_new, f, delta2)
        check = tol > 0 and it % check_every == 0
        if check:
            diff = np.sqrt(diff_sq(u, u_new))
        u, u_new = u_new, u          # swap names, no data is copied
        if check and diff < tol:
            break
    return u, it, diff
