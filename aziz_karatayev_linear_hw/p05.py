import math
import time
import numpy as np

rng = np.random.default_rng(0)


def det_cofactor(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    if n == 1:
        return A[0, 0]

    res = 0.0
    for j in range(n):
        sub = np.delete(A[1:], j, axis=1)
        if j % 2 == 0:
            res += A[0, j] * det_cofactor(sub)
        else:
            res -= A[0, j] * det_cofactor(sub)
    return res


def det_rowreduce(A, tol=1e-10):
    U = np.array(A, dtype=float)
    n = U.shape[0]
    sign = 1.0

    for c in range(n):
        best_row = c
        best_val = abs(U[c, c])
        for i in range(c + 1, n):
            if abs(U[i, c]) > best_val:
                best_val = abs(U[i, c])
                best_row = i

        if best_val < tol:
            return 0.0

        if best_row != c:
            tmp = U[c].copy()
            U[c] = U[best_row]
            U[best_row] = tmp
            sign = -sign

        for i in range(c + 1, n):
            k = U[i, c] / U[c, c]
            U[i, c:] = U[i, c:] - k * U[c, c:]

    det = sign
    for i in range(n):
        det *= U[i, i]
    return det


if __name__ == "__main__":
    M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    F = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("M:", det_cofactor(M), det_rowreduce(M))
    print("F:", det_cofactor(F), det_rowreduce(F))

    A = rng.normal(size=(5, 5))
    B = rng.normal(size=(5, 5))
    print("vs np:", np.isclose(det_cofactor(A), np.linalg.det(A)),
          np.isclose(det_rowreduce(A), np.linalg.det(A)))
    print("det(AB):", np.isclose(det_rowreduce(A @ B), det_rowreduce(A) * det_rowreduce(B)))
    print("det(A.T):", np.isclose(det_rowreduce(A.T), det_rowreduce(A)))

    t8 = 0.0
    for n in range(2, 9):
        X = rng.normal(size=(n, n))
        t0 = time.perf_counter()
        det_cofactor(X)
        dt = time.perf_counter() - t0
        print("cofactor n =", n, "time =", round(dt, 6))
        if n == 8:
            t8 = dt

    est20 = t8 * (math.factorial(20) / math.factorial(8))
    print("est n=20 (years):", est20 / (365 * 24 * 3600))

    Big = rng.normal(size=(200, 200))
    t0 = time.perf_counter()
    det_rowreduce(Big)
    print("rowreduce n=200 time:", round(time.perf_counter() - t0, 5))