import numpy as np

rng = np.random.default_rng(0)


def fib_exact(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def fib_binet(n):
    s5 = np.sqrt(5.0)
    phi = (1.0 + s5) / 2.0
    psi = (1.0 - s5) / 2.0
    return round((phi ** n - psi ** n) / s5)


if __name__ == "__main__":
    A = rng.normal(size=(5, 5))
    vals, P = np.linalg.eig(A)
    D = np.diag(vals)
    P_inv = np.linalg.inv(P)

    print("A == PDP^-1:", np.allclose(A, P @ D @ P_inv))
    print("A^20 ok:", np.allclose(P @ np.diag(vals ** 20) @ P_inv, np.linalg.matrix_power(A, 20)))

    for n in range(1, 100):
        e = fib_exact(n)
        b = fib_binet(n)
        if e != b:
            print("first binet error at n =", n, "exact:", e, "binet:", b)
            break

    S = np.array([[1.0, 1.0], [0.0, 1.0]])
    sv, sp = np.linalg.eig(S)
    print("shear vals:", sv)
    print("shear vecs:\n", sp)