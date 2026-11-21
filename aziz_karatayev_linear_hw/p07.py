import numpy as np
from p05 import det_rowreduce

rng = np.random.default_rng(0)


def is_symmetric(A, tol=1e-10):
    A = np.array(A, dtype=float)
    if A.shape[0] != A.shape[1]:
        return False
    return np.allclose(A, A.T, atol=tol, rtol=0)


def is_orthogonal(Q, tol=1e-10):
    Q = np.array(Q, dtype=float)
    if Q.shape[0] != Q.shape[1]:
        return False
    n = Q.shape[0]
    return np.allclose(Q.T @ Q, np.eye(n), atol=tol, rtol=0)


def trace(A):
    s = 0.0
    for i in range(A.shape[0]):
        s += A[i, i]
    return s


def split_sym_skew(A):
    A = np.array(A, dtype=float)
    S = 0.5 * (A + A.T)
    K = 0.5 * (A - A.T)
    return S, K


if __name__ == "__main__":
    A = rng.normal(size=(3, 5))
    B = rng.normal(size=(5, 3))
    print("A.T @ A sym:", is_symmetric(A.T @ A))
    print("tr(AB) == tr(BA):", np.isclose(trace(A @ B), trace(B @ A)))
    print("tr(A.T A) == fro^2:", np.isclose(trace(A.T @ A), np.linalg.norm(A, "fro") ** 2))

    M = rng.normal(size=(4, 4))
    S, K = split_sym_skew(M)
    print("split ok:", np.allclose(M, S + K), is_symmetric(S), np.allclose(K.T, -K))

    Q, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    print("Q orth:", is_orthogonal(Q))
    print("det(Q) = +-1:", np.isclose(abs(det_rowreduce(Q)), 1.0))

    x = rng.normal(size=4)
    print("norm preserved:", np.isclose(np.linalg.norm(Q @ x), np.linalg.norm(x)))