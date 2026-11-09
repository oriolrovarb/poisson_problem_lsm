"""Problem definition from the assignment: source term, exact solution, boundary.

The field functions take 1D coordinate arrays x, y, z (one per axis) instead of
N, so an MPI rank can pass the coordinates of its own local block.
"""
import numpy as np


def _sin_product(x, y, z):
    """sin(pi x) sin(pi y) sin(pi z) on the 3D block spanned by x, y, z.

    Broadcasting: x[:, None, None] has shape (nx, 1, 1), y[None, :, None] has
    (1, ny, 1), z[None, None, :] has (1, 1, nz); their product is (nx, ny, nz).
    Same result as np.meshgrid(..., indexing="ij") but without three full 3D
    coordinate arrays in memory.
    """
    return (np.sin(np.pi * x)[:, None, None]
            * np.sin(np.pi * y)[None, :, None]
            * np.sin(np.pi * z)[None, None, :])


def source_term(x, y, z):
    """f(x, y, z) = 3 pi^2 sin(pi x) sin(pi y) sin(pi z)."""
    return 3.0 * np.pi**2 * _sin_product(x, y, z)


def exact_solution(x, y, z):
    """u(x, y, z) = sin(pi x) sin(pi y) sin(pi z), used to measure the error."""
    return _sin_product(x, y, z)


def apply_dirichlet(u, value=0.0):
    """Set all six outer faces of a global array to the Dirichlet value (u = 0).

    MPI: on a local block only call this for faces on the physical cube
    boundary; the other faces are ghost layers filled by the halo exchange.
    """
    u[0, :, :] = u[-1, :, :] = value
    u[:, 0, :] = u[:, -1, :] = value
    u[:, :, 0] = u[:, :, -1] = value
