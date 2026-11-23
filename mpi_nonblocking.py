#let's do Isend IRecv to try to solve the scalability issue and the stop at N=12.
#seems that issue is solved with the technique used in mpi_profiling
#In this code it will be separated in 1D and also use isend and irecv to make the system work
"""Entry point: MPI Jacobi solver, 1D slab decomposition along axis 0 (i)."""
from mpi4py import MPI

from poisson.args import parse_args
from poisson.grid import grid_spacing, coordinates, allocate
from poisson.problem import source_term, max_error, apply_dirichlet
from poisson.jacobi import jacobi, jacobi_step, diff_sq
from poisson.metrics import performance
from poisson.output import report, report_ranks
from poisson.decomposition import slab_communicator, slab_range, exchange_halos_nonblocking


def main():
    # 1. run-time parameters, line of ranks and my slices
    args = parse_args()
    N = args.N
    delta = grid_spacing(N)
    x = coordinates(N)

    cart, lower, upper = slab_communicator(MPI.COMM_WORLD)
    rank, P = cart.Get_rank(), cart.Get_size()
    if P > N:
        raise SystemExit(f"need at least one slice per rank: P={P} > N={N}")
    start, stop = slab_range(N, P, rank)
    n_local = stop - start
    x_local = x[start - 1:stop + 1]          # my slices + one ghost layer each side

    # 2. allocate, 3. initialise (ghost layers are overwritten by the halo exchange)
    u, u_new, f = allocate((n_local + 2, N + 2, N + 2))
    f[...] = source_term(x_local, x, x)
    apply_dirichlet(u)
    apply_dirichlet(u_new)

    # Per-rank time split, for the load-balance plot. "comm" includes waiting
    # for neighbours, so an overloaded neighbour shows up as comm time here.
    times = {"compute": 0.0, "comm": 0.0}

    def mpi_step(u, u_new, f, delta2):
        t_a = MPI.Wtime()
        reqs=exchange_halos_nonblocking(cart, u, lower, upper)
        MPI.Request.Waitall(reqs)
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
    error = cart.allreduce(max_error(u, x_local, x, x), op=MPI.MAX)
    rank_rows = cart.gather(dict(N=N, ranks=P, rank=rank, compute_s=times["compute"],
                                 comm_s=times["comm"]), root=0)
    if rank == 0:
        if args.rank_csv:
            report_ranks(rank_rows, args.rank_csv)
        mlups, gbs = performance(N, iterations, seconds)
        report(dict(impl="mpi-slab", N=N, ranks=P, threads=1,
                    iterations=iterations, max_iter=args.max_iter, tol=args.tol,
                    check_every=args.check_every, time_s=seconds, mlups=mlups,
                    gbs=gbs, final_diff=diff, max_error=error),
               args.csv)


if __name__ == "__main__":
    main()
