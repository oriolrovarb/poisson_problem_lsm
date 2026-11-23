> The first version (`jacobi_poisson.py`) used `N` = all points including the boundary.
> Its `N` corresponded to `N + 2` in the convention, e.g. old `N = 32` is the same grid as `N = 30`.

## Hardware peak memory bandwidth

Intel Xeon E5-2650 v4 (2 sockets per node), from Intel ARK:
4 memory channels per socket, DDR4-2400, 8 bytes per transfer.

- one socket: 4 × 2400·10⁶ × 8 B = 76.8 GB/s (matches ARK "Max Memory Bandwidth")
- one node: 2 × 76.8 = 153.6 GB/s

This is the maximum *supported* speed; the DIMMs installed may be slower (not checkable
without root). STREAM gives the achievable bandwidth to compare our solver against.
