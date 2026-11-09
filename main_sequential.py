"""Entry point: sequential Jacobi solver for the 3D Poisson problem.

    python3 main.py -N 30                                   # run to convergence
    python3 main.py -N 200 --max-iter 100 --tol 0 --csv results/baseline.csv
"""
from poisson.args import parse_args
from poisson.grid import grid_spacing, coordinates, allocate
from poisson.problem import source_term, max_error, apply_dirichlet
from poisson.jacobi import jacobi
from poisson.timing import wall_time, performance
from poisson.output import report, run_info, save_slice


def warmup():
    """Run the loop on a tiny problem before timing (JIT compile for Numba)."""
    u, u_new, f = allocate((6, 6, 6))
    jacobi(u, u_new, f, grid_spacing(4), max_iter=2, tol=1e-30)


def main(argv=None):
    # 1. run-time parameters
    args = parse_args(argv)
    N = args.N
    delta = grid_spacing(N)
    x = coordinates(N)

    # 2. allocate, 3. initialise source term and boundary conditions
    u, u_new, f = allocate((N + 2, N + 2, N + 2))
    f[...] = source_term(x, x, x)
    apply_dirichlet(u)
    apply_dirichlet(u_new)

    warmup()

    # 4. Jacobi iterations: the ONLY timed part
    t0 = wall_time()
    u, iterations, diff = jacobi(u, u_new, f, delta, args.max_iter, args.tol,
                                 args.check_every)
    seconds = wall_time() - t0

    # 5. results
    mlups, gbs = performance(N, iterations, seconds)
    error = max_error(u, x, x, x)
    report(dict(impl="numpy-seq", N=N, ranks=1, threads=1,
                iterations=iterations, max_iter=args.max_iter, tol=args.tol,
                check_every=args.check_every, time_s=seconds, mlups=mlups,
                gbs=gbs, final_diff=diff, max_error=error, **run_info()),
           args.csv)
    if args.slice:
        save_slice(u, args.slice)


if __name__ == "__main__":
    main()
