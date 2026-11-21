import numpy as np

rng = np.random.default_rng(0)


def matrix_of(T, n):
    """Return the matrix of the linear transformation T: R^n -> R^m."""
    cols = []
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        cols.append(T(e))

    m = len(cols[0])
    A = np.zeros((m, n), dtype=float)
    for j in range(n):
        for i in range(m):
            A[i, j] = cols[j][i]
    return A


def looks_linear(T, n, trials=100):
    for _ in range(trials):
        u = rng.normal(size=n)
        v = rng.normal(size=n)
        a = rng.normal()
        if not np.allclose(T(u + v), T(u) + T(v)):
            return False
        if not np.allclose(T(a * v), a * T(v)):
            return False
    return True


if __name__ == "__main__":
    T1 = lambda v: np.array([2 * v[0] + 3 * v[1], 4 * v[0] - v[1]])
    print("T1 matrix:\n", matrix_of(T1, 2))

    T2 = lambda v: np.array([v[0] + 2 * v[1] - v[2], 3 * v[0] - v[1] + 4 * v[2]])
    M2 = matrix_of(T2, 3)
    expected_M2 = np.array([[1.0, 2.0, -1.0], [3.0, -1.0, 4.0]])
    print("T2 matrix:\n", M2)
    print("Matches Example 6.11:", np.allclose(M2, expected_M2))

    shift = lambda v: np.array([v[0] + 1, v[1]])
    prod = lambda v: np.array([v[0] * v[1], v[1]])

    print("T1 linear:", looks_linear(T1, 2))
    print("T2 linear:", looks_linear(T2, 3))
    print("(x+1, y) linear:", looks_linear(shift, 2))
    print("(xy, y) linear:", looks_linear(prod, 2))