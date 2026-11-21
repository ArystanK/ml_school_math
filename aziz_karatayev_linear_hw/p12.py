import numpy as np
import matplotlib.pyplot as plt
from p04 import inverse

rng = np.random.default_rng(0)


def dual_basis(P, tol=1e-10):
    return inverse(P, tol=tol)


def vandermonde(nodes):
    n = len(nodes)
    V = np.zeros((n, n))
    for k in range(n):
        for j in range(n):
            V[k, j] = nodes[k] ** j
    return V


def eval_poly(c, t):
    res = np.zeros_like(t, dtype=float)
    for j in range(len(c)):
        res += c[j] * (t ** j)
    return res


if __name__ == "__main__":
    P = rng.normal(size=(5, 5))
    R = dual_basis(P)

    G = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            G[i, j] = np.sum(R[i] * P[:, j])
    print("ri * bj == I:", np.allclose(G, np.eye(5)))

    x = rng.normal(size=5)
    rec = np.zeros(5)
    for i in range(5):
        rec += np.sum(R[i] * x) * P[:, i]
    print("coords ok:", np.allclose(x, rec))

    nodes0 = np.array([-1.0, 0.0, 1.0])
    L0 = inverse(vandermonde(nodes0))
    print("Lagrange [-1, 0, 1]:\n", L0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    t1 = np.linspace(-1.0, 1.0, 9)
    L1 = inverse(vandermonde(t1), tol=1e-14)
    y1 = np.sin(3 * t1)
    c1 = L1 @ y1
    xs1 = np.linspace(-1.0, 1.0, 400)
    ax1.plot(xs1, np.sin(3 * xs1), label="sin(3t)")
    ax1.plot(xs1, eval_poly(c1, xs1), "--", label="p(t)")
    ax1.scatter(t1, y1, color="red")
    ax1.set_title("sin(3t), 9 nodes")
    ax1.legend()

    t2 = np.linspace(-5.0, 5.0, 21)
    L2 = inverse(vandermonde(t2), tol=1e-14)
    y2 = 1.0 / (1.0 + t2 ** 2)
    c2 = L2 @ y2
    xs2 = np.linspace(-5.0, 5.0, 400)
    ax2.plot(xs2, 1.0 / (1.0 + xs2 ** 2), label="1/(1+t^2)")
    ax2.plot(xs2, eval_poly(c2, xs2), "--", label="p(t)")
    ax2.scatter(t2, y2, color="red")
    ax2.set_ylim(-1, 2)
    ax2.set_title("Runge, 21 nodes")
    ax2.legend()

    plt.tight_layout()
    plt.show()