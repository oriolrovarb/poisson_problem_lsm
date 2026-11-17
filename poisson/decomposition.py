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


# ---------------------------------------------------------------------------
# N-dimensional (slab / pencil / cube) decomposition on a Cartesian communicator
# ---------------------------------------------------------------------------
import numpy as np
from mpi4py import MPI


def cart_communicator(comm, dims=None, ndims=3):
    """Create an `ndims`-D Cartesian communicator over all ranks of `comm`.

    dims: process grid (Px, Py, Pz), or None to let MPI.Compute_dims choose
    (balanced, e.g. P=24 -> [4, 3, 2]). Use dims=(P, 1, 1) for slabs,
    (Px, Py, 1) for pencils and (Px, Py, Pz) for cubes. Axis d of the grid
    splits array axis d (0=i, 1=j, 2=k).

    Returns (cart, dims, coords, neighbours) where neighbours is a list of
    (lower, upper) rank pairs, one per axis; MPI.PROC_NULL on the physical
    boundary (non-periodic), so the halo exchange needs no special case.
    reorder=False keeps the rank order of `comm` (placement stays under the
    control of mpirun --map-by / --bind-to).
    """
    P = comm.Get_size()
    if dims is None:
        dims = MPI.Compute_dims(P, ndims)
    dims = list(dims)
    if len(dims) != ndims or int(np.prod(dims)) != P:
        raise ValueError(f"dims {dims} do not multiply to {P} ranks")
    cart = comm.Create_cart(dims=dims, periods=[False] * ndims, reorder=False)
    coords = cart.Get_coords(cart.Get_rank())
    neighbours = [cart.Shift(d, 1) for d in range(ndims)]
    return cart, dims, coords, neighbours


def block_range(N, P_d, coord_d):
    """Global interior indices owned along one axis: (start, stop), stop exclusive.

    Same split as slab_range, applied per axis: global interior indices are
    1..N, first N % P_d blocks get one extra point.
    """
    return slab_range(N, P_d, coord_d)


def exchange_halos_nd(cart, u, neighbours):
    """Fill all ghost faces of u from the neighbours, one axis at a time.

    For axis d, u[1] / u[-2] (along d) are my first / last own faces. Faces
    are sent whole (including the ghost rows of the other axes); because the
    axes are done in order 0, 1, 2, the edge/corner ghosts end up filled too,
    though the 7-point stencil never reads them. Only axis 0 faces are
    contiguous in C order; the others are strided, so they are packed into
    contiguous buffers (np.ascontiguousarray) before sending.
    With MPI.PROC_NULL neighbours, the ghost face is left untouched (it keeps
    its Dirichlet value).
    """
    for d, (lower, upper) in enumerate(neighbours):
        def face(idx):
            sl = [slice(None)] * u.ndim
            sl[d] = idx
            return tuple(sl)

        # upward: my last own face -> upper; lower's last own face -> my ghost 0
        recv = np.empty_like(u[face(0)])
        cart.Sendrecv(sendbuf=np.ascontiguousarray(u[face(-2)]), dest=upper,
                      recvbuf=recv, source=lower)
        if lower != MPI.PROC_NULL:
            u[face(0)] = recv
        # downward: my first own face -> lower; upper's first own face -> my ghost -1
        recv = np.empty_like(u[face(-1)])
        cart.Sendrecv(sendbuf=np.ascontiguousarray(u[face(1)]), dest=lower,
                      recvbuf=recv, source=upper)
        if upper != MPI.PROC_NULL:
            u[face(-1)] = recv
