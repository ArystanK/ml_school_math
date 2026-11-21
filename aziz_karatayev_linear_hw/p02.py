import time
import numpy as np

rng = np.random.default_rng(0)


def matmul(A, B):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    if A.shape[1] != B.shape[0]:
        raise ValueError("incompatible shapes")

    m, n = A.shape
    p = B.shape[1]
    res = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            res[i, j] = s
    return res


if __name__ == "__main__":
    shapes = [((2, 3), (3, 4)), ((4, 4), (4, 4)), ((5, 2), (2, 6))]
    for s1, s2 in shapes:
        A = rng.normal(size=s1)
        B = rng.normal(size=s2)
        print(s1, s2, np.allclose(matmul(A, B), A @ B))

    A = rng.normal(size=(3, 4))
    B = rng.normal(size=(4, 5))
    C = rng.normal(size=(5, 2))
    print("assoc:", np.allclose(matmul(matmul(A, B), C), matmul(A, matmul(B, C))))
    print("transpose:", np.allclose(matmul(A, B).T, matmul(B.T, A.T)))

    P = rng.normal(size=(2, 2))
    Q = rng.normal(size=(2, 2))
    print("PQ != QP:", not np.allclose(matmul(P, Q), matmul(Q, P)))

    for n in [50, 100, 200]:
        X = rng.normal(size=(n, n))
        Y = rng.normal(size=(n, n))

        t1 = time.perf_counter()
        matmul(X, Y)
        dt1 = time.perf_counter() - t1

        t2 = time.perf_counter()
        _ = X @ Y
        dt2 = time.perf_counter() - t2

        print("n =", n, "matmul:", round(dt1, 5), "numpy:", round(dt2, 6))