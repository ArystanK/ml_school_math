import numpy as np

rng = np.random.default_rng(0)


def softmax_naive(z):
    return np.exp(z) / np.exp(z).sum()


def softmax_stable(z):
    z = z - z.max()
    return np.exp(z) / np.exp(z).sum()


def softmax_centered(z):
    z = z - z.mean()
    return np.exp(z) / np.exp(z).sum()


if __name__ == "__main__":
    ok = True
    for _ in range(20):
        z = rng.normal(size=5)
        c = rng.normal() * 50
        if not np.allclose(softmax_stable(z + c), softmax_stable(z)):
            ok = False
    print("shift invariance:", ok)

    z1 = np.array([1.0, 2.0, 3.0])
    z2 = np.array([1001.0, 1002.0, 1003.0])
    print("z=(1,2,3) naive:", softmax_naive(z1))
    print("z=(1,2,3) stable:", softmax_stable(z1))
    print("z=(1,2,3) centered:", softmax_centered(z1))

    print("z=(1001,1002,1003) naive:", softmax_naive(z2))
    print("z=(1001,1002,1003) stable:", softmax_stable(z2))
    print("z=(1001,1002,1003) centered:", softmax_centered(z2))

    z3 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1000.0])
    print("outlier centered:", softmax_centered(z3))
    print("outlier stable:", softmax_stable(z3))