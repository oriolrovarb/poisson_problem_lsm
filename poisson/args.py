from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser(description="3D Poisson problem solved with Jacobi iterations")
    parser.add_argument("-N", type=int, default=10,
                        help="number of interior grid points per axis (arrays are (N+2)^3)")
    parser.add_argument("--max-iter", type=int, default=5000,
                        help="maximum number of Jacobi iterations")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="stop when ||u_new - u||_2 < tol; tol <= 0 disables the check "
                             "and runs exactly --max-iter iterations (use this for timing)")
    parser.add_argument("--check-every", type=int, default=1,
                        help="compute the convergence norm only every k iterations")
    parser.add_argument("--csv", default=None,
                        help="append one result row to this CSV file")
    parser.add_argument("--slice", default=None,
                        help="save the middle z-slice of u to this .npy file")
    return parser.parse_args()

