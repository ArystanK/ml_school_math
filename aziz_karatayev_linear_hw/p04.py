import numpy as np
from p03 import rref

rng = np.random.default_rng(0)


def inverse(A, tol=1e-10):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("not square")

    M = np.hstack([A, np.eye(n)])
    R, _ = rref(M, tol)
    if not np.allclose(R[:, :n], np.eye(n), atol=tol):
        raise np.linalg.LinAlgError("matrix is singular")
    return R[:, n:]


if __name__ == "__main__":
    A1 = np.array([[2, 3], [4, -1]])
    print("ex1:", np.allclose(inverse(A1), np.array([[1, 3], [4, -2]]) / 14))

    A2 = np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1]])
    print("ex2:", np.allclose(inverse(A2), np.array([[0, -1, 1], [1, 1, -1], [-1, 0, 1]])))

    A3 = np.array([[1, 2], [2, 4]])
    try:
        inverse(A3)
    except np.linalg.LinAlgError as err:
        print("singular:", err)

    X = rng.normal(size=(10, 10))
    Xi = inverse(X)
    print("X @ Xi:", np.allclose(X @ Xi, np.eye(10)))
    print("Xi @ X:", np.allclose(Xi @ X, np.eye(10)))
    print("vs np.linalg.inv:", np.allclose(Xi, np.linalg.inv(X)))