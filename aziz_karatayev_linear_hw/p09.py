import numpy as np
from p02 import matmul
from p04 import inverse

rng = np.random.default_rng(0)


def normal_eq(X, y):
    XtX = matmul(X.T, X)
    Xty = matmul(X.T, y.reshape(-1, 1))
    w = matmul(inverse(XtX), Xty)
    return w.flatten()


if __name__ == "__main__":
    x = np.linspace(0, 1, 50)
    y = 2 + 3 * x + rng.normal(0, 0.1, size=50)
    X = np.column_stack([np.ones_like(x), x])

    w = normal_eq(X, y)
    w_ref = np.linalg.lstsq(X, y, rcond=None)[0]
    print("line w:", w)
    print("lstsq: ", w_ref, "ok:", np.allclose(w, w_ref))

    y2 = 1 - 2 * x + 4 * x ** 2 + rng.normal(0, 0.1, size=50)
    X2 = np.column_stack([np.ones_like(x), x, x ** 2])

    w2 = normal_eq(X2, y2)
    w2_ref = np.linalg.lstsq(X2, y2, rcond=None)[0]
    print("parabola w:", w2)
    print("lstsq:     ", w2_ref, "ok:", np.allclose(w2, w2_ref))