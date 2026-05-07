import numpy as np


def main():
    n = 3
    sqrt2 = np.sqrt(2.)
    A = np.array([
        [1., 2, 4],
        [2, 3, 5],
        [4, 5, 6]
    ])
    rows, cols = np.tril_indices(n)

    A_vec = A[rows, cols]
    off_diag_mask = rows != cols
    A_vec[off_diag_mask] *= sqrt2

    print(A_vec)


if __name__ == '__main__':
    main()
