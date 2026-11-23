from numba import njit, prange


@njit(parallel=True)
def jacobi_step(u, u_new, f, delta2):
    n0, n1, n2 = u.shape
    for i in prange(1, n0 - 1):
        for j in range(1, n1 - 1):
            for k in range(1, n2 - 1):
                u_new[i, j, k] = (1/6) * (u[i-1, j, k] + u[i+1, j, k]
                                          + u[i, j-1, k] + u[i, j+1, k]
                                          + u[i, j, k-1] + u[i, j, k+1]
                                          + delta2 * f[i, j, k])



@njit(parallel=True)
def diff_sq(u, u_new):
    n0, n1, n2 = u.shape
    s = 0.0
    for i in prange(1, n0 - 1):
        for j in range(1, n1 - 1):
            for k in range(1, n2 - 1):
                s+= (u_new[i,j,k] - u[i,j,k])**2

    return s

def warmup():
    """Compile both kernels on a tiny array.

    Must be called BEFORE mpi4py is imported: on DTU HPC, compiling a Numba
    function after MPI_Init crashes with a segmentation fault.
    """
    import numpy as np
    a = np.zeros((4, 4, 4))
    jacobi_step(a, a.copy(), a, 1.0)
    diff_sq(a, a.copy())