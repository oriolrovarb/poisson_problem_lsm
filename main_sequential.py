"""Entry point: sequential Jacobi solver for the 3D Poisson problem."""
import time

from poisson.args import parse_args
from poisson.grid import grid_spacing, coordinates, allocate
from poisson.problem import source_term, max_error, apply_dirichlet
from poisson.jacobi import jacobi
from poisson.metrics import performance
from poisson.output import report, save_slice


def warmup(step, diff_sq):
    """Run the loop on a tiny problem before timing (JIT compile for Numba)."""
    u, u_new, f = allocate((6, 6, 6))
    jacobi(u, u_new, f, grid_spacing(4), max_iter=2, tol=1e-30,
           step=step, diff_sq=diff_sq)

def main():
    # 1. run-time parameters
    args = parse_args()
    if args.numba:
        import numba
        from poisson.jacobi_numba import jacobi_step, diff_sq
        numba.set_num_threads(args.threads)
        impl = "numba-seq"
    else:
        from poisson.jacobi import jacobi_step, diff_sq
        impl = "numpy-seq"
    N = args.N
    delta = grid_spacing(N)
    x = coordinates(N)

    # 2. allocate, 3. initialise source term and boundary conditions
    u, u_new, f = allocate((N + 2, N + 2, N + 2))
    f[...] = source_term(x, x, x)
    apply_dirichlet(u)
    apply_dirichlet(u_new)

    warmup(jacobi_step, diff_sq)

    # 4. Jacobi iterations: the ONLY timed part
    t0 = time.perf_counter()
    u, iterations, diff = jacobi(u, u_new, f, delta, args.max_iter, args.tol,
                                 args.check_every, step=jacobi_step, diff_sq=diff_sq)
    seconds = time.perf_counter() - t0

    # 5. results
    mlups, gbs = performance(N, iterations, seconds)
    error = max_error(u, x, x, x)
    report(dict(impl=impl, N=N, ranks=1, threads=args.threads if args.numba else 1,
                iterations=iterations, max_iter=args.max_iter, tol=args.tol,
                check_every=args.check_every, time_s=seconds, mlups=mlups,
                gbs=gbs, final_diff=diff, max_error=error),
           args.csv)
    if args.slice:
        save_slice(u, args.slice)


if __name__ == "__main__":
    main()
