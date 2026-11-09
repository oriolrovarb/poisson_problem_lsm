"""Domain decomposition: split the N interior planes along axis 0 (i) over P ranks."""


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
