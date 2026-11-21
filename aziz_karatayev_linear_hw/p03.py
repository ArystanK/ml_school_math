import numpy as np

rng = np.random.default_rng(0)


def rref(A, tol=1e-10):
    R = np.array(A, dtype=float)
    m, n = R.shape
    pivots = []
    r = 0

    for c in range(n):
        if r == m:
            break

        best_row = r
        best_val = abs(R[r, c])
        for i in range(r + 1, m):
            if abs(R[i, c]) > best_val:
                best_val = abs(R[i, c])
                best_row = i

        if best_val < tol:
            continue

        if best_row != r:
            tmp = R[r].copy()
            R[r] = R[best_row]
            R[best_row] = tmp

        R[r] = R[r] / R[r, c]

        for i in range(m):
            if i != r:
                R[i] = R[i] - R[i, c] * R[r]

        pivots.append(c)
        r += 1

    return R, pivots


def rank(A, tol=1e-10):
    _, pivots = rref(A, tol)
    return len(pivots)


def null_space_basis(A, tol=1e-10):
    R, pivots = rref(A, tol)
    n = R.shape[1]
    basis = []
    for c in range(n):
        if c not in pivots:
            v = np.zeros(n)
            v[c] = 1.0
            for row, pcol in enumerate(pivots):
                v[pcol] = -R[row, c]
            basis.append(v)
    return basis


def col_space_basis(A, tol=1e-10):
    A = np.array(A, dtype=float)
    _, pivots = rref(A, tol)
    return [A[:, c] for c in pivots]


if __name__ == "__main__":
    A1 = np.array([[1, 2, -1], [3, -1, 4]])
    R1, p1 = rref(A1)
    print(R1, p1)

    F = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    RF, pF = rref(F)
    print(RF, pF)

    ok = True
    for _ in range(20):
        M = rng.normal(size=(5, 7))
        rk = rank(M)
        nb = null_space_basis(M)
        for v in nb:
            if not np.allclose(M @ v, 0):
                ok = False
        if rk + len(nb) != 7 or rk != np.linalg.matrix_rank(M):
            ok = False
    print("20 random tests:", ok)

    L = rng.normal(size=(6, 4)) @ rng.normal(size=(4, 8))
    print("6x8 rank:", rank(L), "null dim:", len(null_space_basis(L)))