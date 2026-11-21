import numpy as np
import matplotlib.pyplot as plt
from p05 import det_rowreduce

rng = np.random.default_rng(0)

sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
f_pts = np.array([
    [0.2, 0.1], [0.4, 0.1], [0.4, 0.45], [0.65, 0.45], [0.65, 0.6],
    [0.4, 0.6], [0.4, 0.75], [0.8, 0.75], [0.8, 0.9], [0.2, 0.9],
])


def signed_area(P):
    k = len(P)
    s = 0.0
    for i in range(k):
        x1, y1 = P[i]
        x2, y2 = P[(i + 1) % k]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def plot_transformation(A, ax=None):
    show_now = ax is None
    if show_now:
        _, ax = plt.subplots()

    A = np.array(A, dtype=float)
    sq_c = np.vstack([sq, sq[0]])
    f_c = np.vstack([f_pts, f_pts[0]])

    sq_t = sq_c @ A.T
    f_t = f_c @ A.T

    ax.plot(sq_c[:, 0], sq_c[:, 1], "--", color="gray")
    ax.fill(f_c[:, 0], f_c[:, 1], color="lightgray")

    ax.fill(sq_t[:, 0], sq_t[:, 1], color="lightblue", alpha=0.4)
    ax.plot(sq_t[:, 0], sq_t[:, 1], color="blue")

    for t in [0.25, 0.5, 0.75]:
        v = np.array([[t, 0.0], [t, 1.0]]) @ A.T
        h = np.array([[0.0, t], [1.0, t]]) @ A.T
        ax.plot(v[:, 0], v[:, 1], color="blue", linewidth=0.5, alpha=0.6)
        ax.plot(h[:, 0], h[:, 1], color="blue", linewidth=0.5, alpha=0.6)

    ax.fill(f_t[:, 0], f_t[:, 1], color="orange")

    ax.quiver(0, 0, A[0, 0], A[1, 0], angles="xy", scale_units="xy", scale=1, color="red")
    ax.quiver(0, 0, A[0, 1], A[1, 1], angles="xy", scale_units="xy", scale=1, color="green")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_aspect("equal")

    if show_now:
        plt.show()


if __name__ == "__main__":
    cases = [
        ("stretch", np.array([[2, 0], [0, 1]])),
        ("rot90", np.array([[0, -1], [1, 0]])),
        ("reflect", np.array([[1, 0], [0, -1]])),
        ("shear", np.array([[1, 1], [0, 1]])),
        ("singular", np.array([[1, 2], [2, 4]])),
        ("fig4", np.array([[3, 1], [1, 2]])),
    ]

    for name, M in cases:
        area = signed_area(sq @ M.T)
        d = det_rowreduce(M)
        print(name, "area =", round(area, 4), "det =", round(d, 4), np.isclose(area, d))

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (name, M) in zip(axes.flat, cases):
        plot_transformation(M, ax=ax)
        ax.set_title(name)
    plt.tight_layout()
    plt.show()