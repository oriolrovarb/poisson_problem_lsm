"""Domain decomposition: split the N interior planes along axis 0 (i) over P ranks."""


def slab_communicator(comm):
    """Create a 1D Cartesian communicator over all ranks of `comm`.

    Returns (cart, lower, upper): the new communicator and the ranks of the
    neighbours below (smaller i) and above (larger i). periods=[False]: the
    domain is not periodic, so the first rank has lower = MPI.PROC_NULL and the
    last rank has upper = MPI.PROC_NULL. Communication with MPI.PROC_NULL does
    nothing, so the halo exchange needs no special case for the boundary ranks.
    """
    cart = comm.Create_cart(dims=[comm.Get_size()], periods=[False], reorder=False)
    lower, upper = cart.Shift(0, 1)
    return cart, lower, upper


def slab_range(N, P, rank):
    """Global interior planes owned by `rank`: returns (start, stop), stop exclusive.

    Global interior indices are 1..N. Planes are split as evenly as possible;
    the first N % P ranks get one extra plane.
    Example N=10, P=3: rank 0 -> (1, 5), rank 1 -> (5, 8), rank 2 -> (8, 11).
    """
    base, extra = divmod(N, P)
    n_local = base + (1 if rank < extra else 0)
    start = 1 + rank * base + min(rank, extra)
    return start, start + n_local


def exchange_halos(cart, u, lower, upper):
    """Fill the two ghost slices of u with copies of the neighbours' edge slices.

    u[0] and u[-1] are ghosts; u[1] and u[-2] are this rank's first and last
    own slices. Each slice u[i] is contiguous in memory, so it is sent as is.
    For the first/last rank, lower/upper is MPI.PROC_NULL and that part does
    nothing, so u[0] / u[-1] keep the boundary value.
    """
    # pass upward: my last slice -> rank above, rank below's last slice -> my u[0]
    cart.Sendrecv(sendbuf=u[-2], dest=upper, recvbuf=u[0], source=lower)
    # pass downward: my first slice -> rank below, rank above's first slice -> my u[-1]
    cart.Sendrecv(sendbuf=u[1], dest=lower, recvbuf=u[-1], source=upper)
