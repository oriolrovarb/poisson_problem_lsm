"""Performance metrics, shared by all implementations.
"""

# Bytes moved per lattice update, course convention (Week02 slides):
# 7 loads + 1 store, 8 bytes each (float64).
BYTES_PER_LUP = 64


def performance(N, iterations, seconds):
    """Return (Mlup/s, GB/s) for a run on N^3 interior points.

    lups = N^3 * iterations (one update per interior point per iteration)
    GB/s = Mlup/s * 64 B / 1000, an estimate: loads and stores can overlap,
    and with tol > 0 the convergence check adds traffic that is not counted.
    """
    mlups = N**3 * iterations / seconds / 1e6
    gbs = mlups * BYTES_PER_LUP / 1e3
    return mlups, gbs
