"""Grid helpers. Conventions (N, Δ, index order) are described in conventions.md."""
import numpy as np


def grid_spacing(N):
    """Distance Δ between neighbouring points: N+2 points span [-1, 1]."""
    return 2.0 / (N + 1)


def coordinates(N):
    """1D coordinates of all N+2 points along one axis, boundary included."""
    return np.linspace(-1.0, 1.0, N + 2)


def allocate(shape):
    """Allocate zero-filled u, u_new and f for a padded block of the given shape.

    "Padded" means: the points the block updates, plus one extra layer on each
    side holding the neighbour values the 7-point stencil needs. That outer layer
    is either
      * a boundary layer: fixed Dirichlet values (sequential code, or an MPI rank
        whose face lies on the cube boundary)
      * a ghost layer (MPI only): a copy of the neighbouring rank's edge values,
        refreshed by a halo exchange before every Jacobi sweep, because each rank
        only has its own array and cannot read its neighbour's memory.
    """
    return np.zeros(shape), np.zeros(shape), np.zeros(shape)


