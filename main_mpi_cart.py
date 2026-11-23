"""Entry point: MPI Jacobi solver, 3D Cartesian decomposition (slab, pencil or cube, set by --dims)."""
import sys
if "--numba" in sys.argv:
    # compile the Numba kernels BEFORE mpi4py starts MPI (see jacobi_numba.warmup)
    from poisson.jacobi_numba import warmup
    warmup()
from mpi4py import MPI

from poisson.args import parse_args
from poisson.grid import grid_spacing, coordinates, allocate
from poisson.problem import source_term, max_error, apply_dirichlet_faces
from poisson.jacobi import jacobi, jacobi_step, diff_sq
from poisson.metrics import performance
from poisson.output import report, report_ranks
from poisson.decomposition import cart_communicator, block_range, exchange_halos_nd


def main():
    # 1. run-time parameters, line of ranks and my slices
    args = parse_args()
    if args.numba:
        import numba
        from poisson.jacobi_numba import jacobi_step, diff_sq
        numba.set_num_threads(args.threads)
    else:
        from poisson.jacobi import jacobi_step, diff_sq
    N = args.N
    delta = grid_spacing(N)
    x = coordinates(N)

    cart, dims, coords, neighbours = cart_communicator(MPI.COMM_WORLD, args.dims)
    rank, P = cart.Get_rank(), cart.Get_size()
    if max(dims) > N:
        raise SystemExit(f"need at least one point per rank along each axis: dims={dims} > N={N}")
    ranges = [block_range(N, dims[d], coords[d]) for d in range(3)]
    # my points + one ghost/boundary layer each side, per axis
    xs = [x[a - 1:b + 1] for a, b in ranges]

    # 2. allocate, 3. initialise (ghost layers are overwritten by the halo exchange)
    u, u_new, f = allocate(tuple(b - a + 2 for a, b in ranges))
    f[...] = source_term(*xs)
    on_lower = [n[0] == MPI.PROC_NULL for n in neighbours]
    on_upper = [n[1] == MPI.PROC_NULL for n in neighbours]
    apply_dirichlet_faces(u, on_lower, on_upper)
    apply_dirichlet_faces(u_new, on_lower, on_upper)

    # Per-rank time split, for the load-balance plot. "comm" includes waiting
    # for neighbours, so an overloaded neighbour shows up as comm time here.
    times = {"compute": 0.0, "comm": 0.0}

    def mpi_step(u, u_new, f, delta2):
        t_a = MPI.Wtime()
        exchange_halos_nd(cart, u, neighbours)
        t_b = MPI.Wtime()
        jacobi_step(u, u_new, f, delta2)
        t_c = MPI.Wtime()
        times["comm"] += t_b - t_a
        times["compute"] += t_c - t_b

    def mpi_diff_sq(u, u_new):
        t_a = MPI.Wtime()
        local = diff_sq(u, u_new)
        t_b = MPI.Wtime()
        total = cart.allreduce(local, op=MPI.SUM)
        t_c = MPI.Wtime()
        times["compute"] += t_b - t_a
        times["comm"] += t_c - t_b
        return total

    # 4. Jacobi iterations: the ONLY timed part; all ranks start together
    cart.Barrier()
    t0 = MPI.Wtime()
    u, iterations, diff = jacobi(u, u_new, f, delta, args.max_iter, args.tol,
                                 args.check_every, step=mpi_step, diff_sq=mpi_diff_sq)
    seconds = cart.allreduce(MPI.Wtime() - t0, op=MPI.MAX)   # slowest rank

    # 5. results: combine over ranks, rank 0 reports
    error = cart.allreduce(max_error(u, *xs), op=MPI.MAX)
    rank_rows = cart.gather(dict(N=N, ranks=P, rank=rank, compute_s=times["compute"],
                                 comm_s=times["comm"]), root=0)
    if rank == 0:
        if args.rank_csv:
            report_ranks(rank_rows, args.rank_csv)
        n_split = sum(d > 1 for d in dims)
        kind = {0: "slab", 1: "slab", 2: "pencil", 3: "cube"}[n_split]
        prefix = "hybrid" if args.numba else "mpi"
        impl = f"{prefix}-{kind}-{'x'.join(map(str, dims))}"
        mlups, gbs = performance(N, iterations, seconds)
        report(dict(impl=impl, N=N, ranks=P, threads=args.threads if args.numba else 1,
                    iterations=iterations, max_iter=args.max_iter, tol=args.tol,
                    check_every=args.check_every, time_s=seconds, mlups=mlups,
                    gbs=gbs, final_diff=diff, max_error=error),
               args.csv)


if __name__ == "__main__":
    main()
