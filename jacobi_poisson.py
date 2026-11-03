import numpy as np

# Grid and domain setup
N = 32
max_iter = 5000
tol = 1e-6

delta = 2.0 / (N - 1)
x = np.linspace(-1, 1, N)
X, Y, Z = np.meshgrid(x, x, x, indexing="ij")

# Source term and exact solution
f = 3 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)

u = np.zeros((N, N, N)) #boundary conditions

# Jacobi iterations
for it in range(1, max_iter + 1):
    u_new = u.copy()

    # 7-point stencil update for inner grid points
    u_new[1:-1, 1:-1, 1:-1] = (1/6.0)*(
        u[:-2, 1:-1, 1:-1]
        + u[2:, 1:-1, 1:-1]
        + u[1:-1, :-2, 1:-1]
        + u[1:-1, 2:, 1:-1]
        + u[1:-1, 1:-1, :-2]
        + u[1:-1, 1:-1, 2:]
        + delta**2 * f[1:-1, 1:-1, 1:-1])

    # Convergence check (Frobenius norm of the change)
    diff = np.linalg.norm(u_new - u)
    u = u_new

    if diff < tol:
        break

max_error = np.max(np.abs(u - u_exact))
print(f"Iterations: {it} | Final diff: {diff:.2e} | Max error: {max_error:.2e}")

#all the profiling must be related with the idea that for better accuracy we need smaller grids
#which means that that the computational time exponentially grows.