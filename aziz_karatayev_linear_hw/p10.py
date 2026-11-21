import numpy as np

rng = np.random.default_rng(0)


def power_iteration(A, num_iter=100):
    x = rng.normal(size=A.shape[0])
    for _ in range(num_iter):
        x = A @ x
        x = x / np.abs(x).max()
    Ax = A @ x
    idx = np.argmax(np.abs(x))
    lam = Ax[idx] / x[idx]
    return lam, x


if __name__ == "__main__":
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    lam, v = power_iteration(A)
    print("ex 10.7:", lam, np.linalg.eig(A)[0])

    J = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    lam, v = power_iteration(J)
    print("ex 10.8:", lam, np.linalg.eig(J)[0])

    R = np.array([[0.0, -1.0], [1.0, 0.0]])
    D = np.array([[1.0, 0.0], [0.0, -1.0]])

    x = rng.normal(size=2)
    print("rot90 steps:")
    for i in range(6):
        x = R @ x
        x = x / np.abs(x).max()
        print(i, x)

    x = rng.normal(size=2)
    print("diag(1,-1) steps:")
    for i in range(6):
        x = D @ x
        x = x / np.abs(x).max()
        print(i, x)

    L = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.5, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    G = 0.85 * L + (0.15 / 4.0) * np.ones((4, 4))

    lam, v = power_iteration(G, num_iter=200)
    pr = v / v.sum()
    print("pagerank lam:", lam)
    print("pagerank:", pr)
    print("G @ pr == pr:", np.allclose(G @ pr, pr))
    print("order:", (np.argsort(-pr) + 1).tolist())