import numpy as np
from p04 import inverse

rng = np.random.default_rng(0)


def hilbert(n):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1)
    return H


if __name__ == "__main__":
    for n in [4, 6, 8, 10, 12]:
        H = hilbert(n)
        x_true = np.ones(n)
        b = H @ x_true

        x1 = np.linalg.solve(H, b)
        x2 = np.linalg.inv(H) @ b

        print("n =", n)
        print("solve: err =", np.max(np.abs(x1 - 1)), "res =", np.max(np.abs(H @ x1 - b)))
        print("inv:   err =", np.max(np.abs(x2 - 1)), "res =", np.max(np.abs(H @ x2 - b)))

        try:
            x3 = inverse(H) @ b
            print("myinv: err =", np.max(np.abs(x3 - 1)), "res =", np.max(np.abs(H @ x3 - b)))
        except np.linalg.LinAlgError as err:
            print("myinv failed:", err)