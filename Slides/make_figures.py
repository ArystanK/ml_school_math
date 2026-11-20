"""Draw the data figures of the slides (simulations, real data, 3D pictures) into Slides/figures/.

    python Slides/make_figures.py            all figures
    python Slides/make_figures.py s06 s08    only the figures of these decks

Needs NumPy and Matplotlib. The figures are committed, so the decks compile without Python.
Every figure is drawn at the size it has on the slide (the text width of a 3:2 slide is
about 5.6 inches), so the fonts match the slides. The random seeds are fixed.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
W = 5.6  # text width of a slide, in inches

ACCENT = "#1f4e8c"
GREEN = "#186e5c"
ORANGE = "#c0601a"
PURPLE = "#782878"
RED = "#b02a2a"
GRAY = "#888888"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "mathtext.fontset": "cm",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

FIGURES = {}


def figure(deck):
    def register(func):
        FIGURES[func.__name__] = (deck, func)
        return func
    return register


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Lecture 5: eigenvalues and tensors

@figure("s05")
def s05_powers():
    """Powers of a matrix: A^k x turns towards the dominant eigenvector; a Markov chain converges like 0.4^k."""
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.1), gridspec_kw={"width_ratios": [1, 1.35]})
    ax = axes[0]
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color="#dddddd", linewidth=0.8)
    x = np.array([1.0, -0.6])
    cmap = plt.get_cmap("Blues")
    for k in range(6):
        v = np.linalg.matrix_power(A, k) @ x
        v = v / np.linalg.norm(v)
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=cmap(0.35 + 0.13 * k), linewidth=1.3))
        if k in (0, 5):
            ax.text(*(1.2 * v + np.array([0.1, -0.05])), r"$\mathbf{x}$" if k == 0 else r"$A^5\mathbf{x}$", fontsize=8,
                    ha="center", va="center", color=cmap(0.4 + 0.12 * k))
    q = np.array([1, 1]) / np.sqrt(2)
    ax.plot(*np.c_[-1.3 * q, 1.3 * q], color=RED, linewidth=0.8, linestyle="--")
    ax.text(0.95, 1.18, r"$E_3$", color=RED, fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-1.4, 1.5)
    ax.set_ylim(-1.3, 1.4)
    ax.set_axis_off()
    ax.set_title(r"$A^k\mathbf{x}$, $k = 0, \ldots, 5$ (normalized)", fontsize=8)
    ax = axes[1]
    P = np.array([[0.9, 0.5], [0.1, 0.5]])
    for x0, c, lab in [(np.array([1.0, 0.0]), ORANGE, "start sunny"), (np.array([0.0, 1.0]), ACCENT, "start rainy")]:
        xs = [x0]
        for _ in range(12):
            xs.append(P @ xs[-1])
        ax.plot(np.arange(13), [v[0] for v in xs], "o-", color=c, markersize=3, linewidth=1, label=lab)
    ax.axhline(5 / 6, color=GRAY, linestyle=":", linewidth=0.9)
    ax.text(12, 5 / 6 + 0.04, r"$\frac{5}{6}$", ha="right", fontsize=9, color=GRAY)
    ax.set_xlabel("day $k$", labelpad=1)
    ax.set_title(r"$P(\text{sunny on day } k)$: error $\propto 0.4^k$", fontsize=8)
    ax.set_ylim(-0.05, 1.08)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.tick_params(labelsize=7)
    save(fig, "s05_powers")


@figure("s05")
def s05_convolution():
    """A circulant smoothing matrix, its action on a noisy signal, and its eigenvalues = the DFT of the filter."""
    n = 32
    c = np.zeros(n)
    c[[0, 1, 2, -1, -2]] = [0.4, 0.2, 0.1, 0.2, 0.1]
    C = np.array([[c[(j - k) % n] for k in range(n)] for j in range(n)])
    rng = np.random.default_rng(14)
    j = np.arange(n)
    x = np.sin(2 * np.pi * j / n) + 0.5 * np.sin(2 * np.pi * 13 * j / n) + rng.normal(0, 0.15, n)
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.8), gridspec_kw={"width_ratios": [0.9, 1.3, 1.1]})
    ax = axes[0]
    ax.imshow(C, cmap="Blues")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"$\mathrm{circ}(\mathbf{c})$, $n = 32$", fontsize=8)
    ax = axes[1]
    ax.plot(j, x, ".-", color=GRAY, linewidth=0.7, markersize=3, label=r"$\mathbf{x}$")
    ax.plot(j, C @ x, ".-", color=ACCENT, linewidth=1.1, markersize=3, label=r"$\mathbf{c} \ast \mathbf{x}$")
    ax.set_title("smoothing a signal", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="upper right", ncol=2)
    ax.set_ylim(-2.2, 2.4)
    ax.tick_params(labelsize=7)
    ax = axes[2]
    chat = np.fft.fft(c).real
    ax.bar(np.arange(n), chat, color=ACCENT, width=0.7)
    ax.axhline(0, color=GRAY, linewidth=0.6)
    ax.set_title(r"eigenvalues $\hat c_m$ (frequency $m$)", fontsize=8)
    ax.set_xlabel("$m$", labelpad=1)
    ax.tick_params(labelsize=7)
    save(fig, "s05_convolution")


# ---------------------------------------------------------------- Lecture 6: linear algebra II

@figure("s06")
def s06_projection3d():
    """Projection of x onto a plane U: the residual x - pi_U(x) is orthogonal to U."""
    x = np.array([1.6, 1.2, 1.8])
    p = np.array([1.6, 1.2, 0.0])
    fig = plt.figure(figsize=(3.0, 2.5))
    ax = fig.add_subplot(projection="3d")
    s_, t_ = np.meshgrid(np.linspace(-0.4, 2.6, 2), np.linspace(-0.4, 2.2, 2))
    ax.plot_surface(s_, t_, 0 * s_, color=ACCENT, alpha=0.13, linewidth=0)
    arrows = [(np.array([1.0, 0, 0]), GRAY, r"$\mathbf{b}_1$"), (np.array([0, 1.0, 0]), GRAY, r"$\mathbf{b}_2$"),
              (x, RED, r"$\mathbf{x}$"), (p, ACCENT, r"$\pi_U(\mathbf{x})$")]
    for v, c, lab in arrows:
        ax.quiver(0, 0, 0, *v, color=c, arrow_length_ratio=0.12, linewidth=1.5)
        ax.text(*(v * 1.08 + np.array([0, 0, 0.08])), lab, color=c, fontsize=9)
    ax.plot(*np.c_[p, x], color=GREEN, linestyle="--", linewidth=1.3)
    ax.text(*(np.array([1.65, 1.25, 0.9])), r"$\ \mathbf{x} - \pi_U(\mathbf{x})$", color=GREEN, fontsize=9)
    ax.text(2.2, -0.3, 0, r"$U$", color=ACCENT, fontsize=10)
    ax.set_xlim(-0.4, 2.6)
    ax.set_ylim(-0.4, 2.2)
    ax.set_zlim(0, 2.0)
    ax.set_box_aspect((3, 2.6, 2))
    ax.view_init(elev=20, azim=-62)
    ax.set_axis_off()
    save(fig, "s06_projection3d")


@figure("s06")
def s06_lowrank():
    """Rank-k approximations of a terrain elevation map, and the decay of its singular values."""
    elevation = cbook.get_sample_data("jacksboro_fault_dem.npz")["elevation"].astype(float)
    U, s, Vt = np.linalg.svd(elevation, full_matrices=False)
    m, n = elevation.shape
    fig, axes = plt.subplots(1, 5, figsize=(W, 1.45), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.15]})
    vmin, vmax = elevation.min(), elevation.max()
    for ax, k in zip(axes[:4], [1, 5, 20, None]):
        A = elevation if k is None else (U[:, :k] * s[:k]) @ Vt[:k]
        ax.imshow(A, cmap="gist_earth", vmin=vmin, vmax=vmax)
        if k is None:
            ax.set_title("original", fontsize=8)
            ax.set_xlabel(f"{m * n} numbers", fontsize=7)
        else:
            err = np.sqrt((s[k:] ** 2).sum() / (s ** 2).sum())
            ax.set_title(f"$k = {k}$, error ${100 * err:.0f}\\%$", fontsize=8)
            ax.set_xlabel(f"{k * (m + n + 1)} numbers", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax = axes[4]
    ax.semilogy(np.arange(1, 61), s[:60] / s[0], ".", color=ACCENT, markersize=3)
    ax.set_yticks([1, 0.1, 0.01], ["$1$", "$0.1$", "$0.01$"])
    ax.minorticks_off()
    ax.set_title(r"$\sigma_i / \sigma_1$", fontsize=8)
    ax.set_xlabel("$i$", labelpad=1)
    ax.tick_params(labelsize=7)
    save(fig, "s06_lowrank")


@figure("s06")
def s06_pca():
    """PCA of a 2D Gaussian cloud: principal directions, and the projection onto the first one."""
    rng = np.random.default_rng(1)
    C = np.array([[3.0, 1.6], [1.6, 1.3]])
    X = rng.multivariate_normal([0, 0], C, size=60)
    X -= X.mean(axis=0)
    S = X.T @ X / len(X)
    lam, Q = np.linalg.eigh(S)
    lam, Q = lam[::-1], Q[:, ::-1]
    q1 = Q[:, 0] * np.sign(Q[0, 0])
    q2 = Q[:, 1] * np.sign(Q[1, 1])
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.35))
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(-4.6, 4.6)
        ax.set_ylim(-3.1, 3.1)
        ax.axhline(0, color="#dddddd", linewidth=0.6, zorder=0)
        ax.axvline(0, color="#dddddd", linewidth=0.6, zorder=0)
        ax.tick_params(labelsize=7)
    ax = axes[0]
    ax.scatter(*X.T, s=7, color=GRAY)
    for q, l, c, name in [(q1, lam[0], RED, r"$\mathbf{q}_1$"), (q2, lam[1], GREEN, r"$\mathbf{q}_2$")]:
        v = 2 * np.sqrt(l) * q
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, linewidth=1.6))
        ax.text(*(v * 1.18), name, color=c, ha="center", va="center", fontsize=9)
    t = np.linspace(0, 2 * np.pi, 200)
    ell = (Q @ np.diag(2 * np.sqrt(lam)) @ np.vstack([np.cos(t), np.sin(t)]))
    ax.plot(*ell, color=ACCENT, linewidth=0.8)
    ax.set_title(r"eigenvectors of $S$ (length $2\sqrt{\lambda_i}$)")
    ax = axes[1]
    proj = np.outer(X @ q1, q1)
    for xi, pi in zip(X, proj):
        ax.plot(*np.c_[xi, pi], color="#bbbbbb", linewidth=0.6)
    ax.plot(*np.c_[-5 * q1, 5 * q1], color=RED, linewidth=1)
    ax.scatter(*X.T, s=7, color=GRAY)
    ax.scatter(*proj.T, s=7, color=RED, zorder=3)
    explained = lam[0] / lam.sum()
    ax.set_title(f"projection onto $\\mathbf{{q}}_1$: ${100 * explained:.0f}\\%$ of the variance")
    save(fig, "s06_pca")


@figure("s06")
def s06_rayleigh():
    """The Rayleigh quotient on the unit circle: maximum and minimum at the eigenvectors."""
    A = np.array([[2.5, 1.5], [1.5, 2.5]])  # eigenvalues 4 and 1, eigenvectors (1,1)/sqrt2, (1,-1)/sqrt2
    th = np.linspace(0, np.pi, 400)
    X = np.vstack([np.cos(th), np.sin(th)])
    R = np.einsum("in,ij,jn->n", X, A, X)
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.9), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    t = np.linspace(0, 2 * np.pi, 300)
    circle = np.vstack([np.cos(t), np.sin(t)])
    ax.plot(*circle, color=GRAY, linewidth=0.8, label=r"$\|\mathbf{x}\| = 1$")
    ax.plot(*(A @ circle), color=ACCENT, linewidth=1.2, label=r"$A\mathbf{x}$")
    for v, c, lab in [(np.array([1, 1]) / np.sqrt(2), RED, r"$\mathbf{q}_1$"), (np.array([1, -1]) / np.sqrt(2), GREEN, r"$\mathbf{q}_2$")]:
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, linewidth=1.4))
        ax.text(*(v * 1.3), lab, color=c, ha="center", va="center")
    ax.set_aspect("equal")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.legend(loc="lower right", frameon=False, fontsize=7, handlelength=1.2)
    ax.set_title(r"$A$ maps the circle to an ellipse")
    ax.tick_params(labelsize=7)
    ax = axes[1]
    ax.plot(np.degrees(th), R, color=ACCENT)
    ax.axhline(4, color=RED, linestyle=":", linewidth=0.8)
    ax.axhline(1, color=GREEN, linestyle=":", linewidth=0.8)
    ax.plot([45], [4], "o", color=RED, markersize=4)
    ax.plot([135], [1], "o", color=GREEN, markersize=4)
    ax.text(47, 3.75, r"$\lambda_1 = 4$ at $\mathbf{q}_1$", color=RED, fontsize=8)
    ax.text(100, 1.25, r"$\lambda_2 = 1$ at $\mathbf{q}_2$", color=GREEN, fontsize=8)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel(r"angle of $\mathbf{x} = (\cos\theta, \sin\theta)$ in degrees", labelpad=1)
    ax.set_title(r"$R_A(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x}$ on the unit circle")
    ax.set_ylim(0.5, 4.5)
    save(fig, "s06_rayleigh")


@figure("s06")
def s06_condition():
    """An ill-conditioned 2x2 system (kappa ~ 42): almost parallel lines, and a small change of b moves the solution."""
    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    x1 = np.linspace(-0.8, 2.8, 2)
    ax.plot(x1, 2 - x1, color=ACCENT, linewidth=1.3, label=r"$x_1 + x_2 = 2$")
    ax.plot(x1, (2.1 - x1) / 1.1, color=GREEN, linewidth=1.3, linestyle="--", label=r"$x_1 + 1.1\,x_2 = 2.1$")
    ax.plot(x1, (2.2 - x1) / 1.1, color=ORANGE, linewidth=1.3, linestyle=":", label=r"$x_1 + 1.1\,x_2 = 2.2$")
    ax.plot([1], [1], "o", color=GREEN, markersize=5)
    ax.plot([0], [2], "o", color=ORANGE, markersize=5)
    ax.annotate(r"$\mathbf{x} = (1, 1)$", (1, 1), xytext=(1.5, 1.45), color=GREEN, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=GREEN, linewidth=0.6))
    ax.annotate(r"$\mathbf{x} = (0, 2)$", (0, 2), xytext=(0.5, 2.45), color=ORANGE, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=0.6))
    ax.set_xlim(-0.8, 2.8)
    ax.set_ylim(-0.8, 2.9)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", frameon=False, fontsize=7, handlelength=1.8)
    ax.tick_params(labelsize=7)
    save(fig, "s06_condition")


@figure("s06")
def s06_spectral_clustering():
    """Spectral clustering: a k-nearest-neighbor graph of two rings, colored by the signs of the Fiedler vector."""
    rng = np.random.default_rng(3)
    n = 120
    t = rng.uniform(0, 2 * np.pi, n)
    r = np.where(np.arange(n) < n // 2, 1.0, 2.4) + rng.normal(0, 0.12, n)
    X = np.c_[r * np.cos(t), r * np.sin(t)]
    D2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    k = 7
    W_ = np.zeros((n, n))
    nbrs = np.argsort(D2, axis=1)[:, 1:k + 1]
    for i in range(n):
        W_[i, nbrs[i]] = 1
    W_ = np.maximum(W_, W_.T)
    L = np.diag(W_.sum(1)) - W_
    lam, V = np.linalg.eigh(L)
    f = V[:, 1]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ii, jj = np.nonzero(np.triu(W_))
    for i, j in zip(ii, jj):
        ax.plot(*np.c_[X[i], X[j]], color="#cccccc", linewidth=0.5, zorder=0)
    ax.scatter(*X.T, s=9, c=np.where(f > 0, RED, ACCENT), zorder=2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(r"$k$-NN graph, colored by $\mathrm{sign}(f_i)$")
    ax = axes[1]
    order = np.argsort(f)
    ax.plot(np.arange(n), f[order], ".", markersize=3, color=PURPLE)
    ax.axhline(0, color=GRAY, linewidth=0.6)
    ax.set_title(r"Fiedler vector $\mathbf{f}$ (sorted), $\lambda_2 = %.3f$" % lam[1])
    ax.set_xlabel("vertices, sorted by $f_i$", labelpad=1)
    ax.tick_params(labelsize=7)
    save(fig, "s06_spectral_clustering")


# ---------------------------------------------------------------- Lecture 7: calculus I

@figure("s07")
def s07_landscape():
    """A two-variable function: its graph, and its contour plot with gradients orthogonal to the level curves."""
    def f(x, y):
        return np.sin(x) * np.cos(y) + 0.08 * (x ** 2 + y ** 2)

    def grad(x, y):
        return np.array([np.cos(x) * np.cos(y) + 0.16 * x, -np.sin(x) * np.sin(y) + 0.16 * y])

    x = np.linspace(-3.5, 3.5, 160)
    X, Y = np.meshgrid(x, x)
    Z = f(X, Y)
    fig = plt.figure(figsize=(W, 2.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, rstride=3, cstride=3, alpha=0.95)
    ax.contour(X, Y, Z, levels=14, offset=Z.min() - 0.6, cmap="viridis", linewidths=0.6)
    ax.set_zlim(Z.min() - 0.6, Z.max())
    ax.view_init(elev=32, azim=-55)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(r"graph of $f(x, y)$")
    ax = fig.add_subplot(1, 2, 2)
    ax.contour(X, Y, Z, levels=14, cmap="viridis", linewidths=0.8)
    pts = np.array([[-2.2, 0.6], [-1.0, -1.6], [0.4, 1.4], [1.6, -0.5], [2.6, 2.2], [-2.5, -2.6], [0.9, 0.2], [-0.4, 2.8]])
    for p in pts:
        g = grad(*p)
        g = 0.9 * g / np.linalg.norm(g)
        ax.annotate("", xy=p + g, xytext=p, arrowprops=dict(arrowstyle="-|>", color=RED, linewidth=1.2, mutation_scale=7))
        ax.plot(*p, "o", color=RED, markersize=2.5)
    ax.set_aspect("equal")
    ax.set_title(r"level curves and $\nabla f$ (normalized)")
    ax.tick_params(labelsize=7)
    save(fig, "s07_landscape")


@figure("s07")
def s07_finite_differences():
    """Errors of forward and central differences: O(h) and O(h^2) until rounding errors take over."""
    h = np.logspace(-15, 0, 200)
    x = 1.0
    fwd = np.abs((np.exp(x + h) - np.exp(x)) / h - np.exp(x))
    cen = np.abs((np.exp(x + h) - np.exp(x - h)) / (2 * h) - np.exp(x))
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ax.loglog(h, fwd, color=ACCENT, label=r"forward $\frac{f(x+h) - f(x)}{h}$")
    ax.loglog(h, cen, color=ORANGE, label=r"central $\frac{f(x+h) - f(x-h)}{2h}$")
    ax.loglog(h[h > 1e-6], h[h > 1e-6], ":", color=ACCENT, linewidth=0.8)
    ax.loglog(h[h > 1e-4], h[h > 1e-4] ** 2, ":", color=ORANGE, linewidth=0.8)
    ax.text(3e-3, 1e-1, r"$\sim h$", color=ACCENT)
    ax.text(1e-2, 2e-6, r"$\sim h^2$", color=ORANGE)
    ax.text(3e-15, 1e-10, "rounding errors", color=GRAY, fontsize=7)
    ax.set_xlabel("$h$", labelpad=1)
    ax.set_ylabel("error for $f = e^x$ at $x = 1$")
    ax.set_ylim(1e-12, 10)
    ax.legend(frameon=False, loc="upper center", fontsize=7)
    ax.tick_params(labelsize=7)
    save(fig, "s07_finite_differences")


@figure("s07")
def s07_vanishing():
    """Norms of the error signals delta_l in a deep random network: sigmoid vanishes, ReLU with He initialization does not."""
    rng = np.random.default_rng(0)
    L, n = 30, 200
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    configs = [
        ("sigmoid, $W_{ij} \\sim N(0, 1/n)$", lambda z: 1 / (1 + np.exp(-z)), lambda z: (s := 1 / (1 + np.exp(-z))) * (1 - s), 1.0, ACCENT),
        ("tanh, $N(0, 1/n)$", np.tanh, lambda z: 1 - np.tanh(z) ** 2, 1.0, GREEN),
        ("ReLU, $N(0, 2/n)$ (He)", lambda z: np.maximum(z, 0), lambda z: (z > 0).astype(float), 2.0, RED),
    ]
    for label, act, dact, gain, color in configs:
        norms = np.zeros(L)
        for trial in range(5):
            Ws = [rng.normal(0, np.sqrt(gain / n), (n, n)) for _ in range(L)]
            a = rng.normal(0, 1, n)
            zs = []
            for Wl in Ws:
                z = Wl @ a
                zs.append(z)
                a = act(z)
            delta = rng.normal(0, 1, n)
            delta /= np.linalg.norm(delta)
            out = [np.linalg.norm(delta)]
            delta = dact(zs[-1]) * delta
            for l in range(L - 1, 0, -1):
                delta = dact(zs[l - 1]) * (Ws[l].T @ delta)
                out.append(np.linalg.norm(delta))
            norms += np.log10(np.array(out[::-1]))
        ax.plot(np.arange(1, L + 1), 10 ** (norms / 5), color=color, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("layer $l$", labelpad=1)
    ax.set_ylabel(r"$\|\boldsymbol{\delta}_l\|$ (output: $1$)")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.tick_params(labelsize=7)
    save(fig, "s07_vanishing")


@figure("s07")
def s07_euler():
    """Euler's method: for x' = -x, and gradient descent as Euler's method for the gradient flow of a quadratic."""
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.2), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    t = np.linspace(0, 3, 200)
    ax.plot(t, np.exp(-t), color="black", linewidth=1.2, label="exact $e^{-t}$")
    for h, c in [(0.5, ORANGE), (0.1, ACCENT)]:
        k = np.arange(0, int(3 / h) + 1)
        ax.plot(k * h, (1 - h) ** k, "o-", color=c, markersize=2.5, linewidth=0.8, label=f"Euler, $h = {h}$")
    ax.set_xlabel("$t$", labelpad=1)
    ax.set_title(r"$x' = -x$, $x(0) = 1$")
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(labelsize=7)
    ax = axes[1]
    lam = np.array([1.0, 10.0])
    xg = np.linspace(-2.2, 2.2, 200)
    XX, YY = np.meshgrid(xg, np.linspace(-1.2, 1.2, 120))
    ax.contour(XX, YY, 0.5 * (lam[0] * XX ** 2 + lam[1] * YY ** 2), levels=[0.1, 0.4, 1, 2, 3.5, 5.5], colors="#cccccc", linewidths=0.6)
    x0 = np.array([-2.0, 1.0])
    tt = np.linspace(0, 6, 300)
    flow = x0[None] * np.exp(-np.outer(tt, lam))
    ax.plot(*flow.T, color="black", linewidth=1.2, label="gradient flow")
    for h, c in [(0.05, ACCENT), (0.19, ORANGE)]:
        xs = [x0]
        for _ in range(40):
            xs.append(xs[-1] * (1 - h * lam))
        xs = np.array(xs)
        ax.plot(*xs.T, "o-", color=c, markersize=2, linewidth=0.7, label=f"GD, $\\eta = {h}$")
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 0.6)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title(r"GD on $\frac{1}{2}(x^2 + 10y^2)$: stable iff $\eta < 0.2$")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, handlelength=1.5, columnspacing=0.8)
    ax.tick_params(labelsize=7)
    save(fig, "s07_euler")


# ---------------------------------------------------------------- Lecture 8: optimization

@figure("s08")
def s08_momentum():
    """Gradient descent, heavy-ball momentum and Newton on an ill-conditioned quadratic (kappa = 25)."""
    lam = np.array([1.0, 25.0])
    x0 = np.array([-4.0, 1.0])
    kappa = lam[1] / lam[0]
    eta_gd = 2 / (lam[0] + lam[1])
    eta_hb = 4 / (np.sqrt(lam[0]) + np.sqrt(lam[1])) ** 2
    beta = ((np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)) ** 2
    K = 60
    gd = [x0]
    for _ in range(K):
        gd.append(gd[-1] - eta_gd * lam * gd[-1])
    hb = [x0, x0 - eta_hb * lam * x0]
    for _ in range(K - 1):
        hb.append(hb[-1] - eta_hb * lam * hb[-1] + beta * (hb[-1] - hb[-2]))
    gd, hb = np.array(gd), np.array(hb)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    X, Y = np.meshgrid(np.linspace(-4.5, 1, 200), np.linspace(-1.6, 1.6, 120))
    ax.contour(X, Y, 0.5 * (lam[0] * X ** 2 + lam[1] * Y ** 2), levels=[0.2, 1, 3, 6, 10, 15], colors="#cccccc", linewidths=0.6)
    ax.plot(*gd[:25].T, "o-", color=ACCENT, markersize=2, linewidth=0.7, label="gradient descent")
    ax.plot(*hb[:25].T, "o-", color=ORANGE, markersize=2, linewidth=0.7, label="momentum")
    ax.plot([x0[0], 0], [x0[1], 0], "o--", color=RED, markersize=3, linewidth=0.9, label="Newton (one step)")
    ax.set_aspect("equal")
    ax.set_xlim(-4.5, 1)
    ax.set_ylim(-1.6, 1.6)
    ax.set_title(r"$f = \frac{1}{2}(x^2 + 25y^2)$, $\kappa = 25$")
    ax.tick_params(labelsize=7)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=7.5, loc="lower center",
               bbox_to_anchor=(0.5, -0.06), ncol=3, handlelength=1.6, columnspacing=1.5)
    ax = axes[1]
    k = np.arange(K + 1)
    ax.semilogy(k, np.linalg.norm(gd, axis=1), color=ACCENT)
    ax.semilogy(k, np.linalg.norm(hb, axis=1), color=ORANGE)
    ax.text(30, 3e-1, r"$\left(\frac{\kappa - 1}{\kappa + 1}\right)^k$", color=ACCENT, fontsize=8)
    ax.text(30, 1e-6, r"$\approx\left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^k$", color=ORANGE, fontsize=8)
    ax.set_xlabel("iteration $k$", labelpad=1)
    ax.set_title(r"error $\|\mathbf{x}_k - \mathbf{x}^*\|$")
    ax.set_ylim(1e-8, 10)
    ax.tick_params(labelsize=7)
    fig.subplots_adjust(bottom=0.2)
    save(fig, "s08_momentum")


@figure("s08")
def s08_lasso():
    """Lasso vs ridge: the constrained problems (l1 diamond vs l2 disk) and soft thresholding."""
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.95), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    w_ls = np.array([2.0, 0.9])
    H = np.array([[1.0, 0.45], [0.45, 0.6]])
    X, Y = np.meshgrid(np.linspace(-1.5, 3.2, 300), np.linspace(-1.5, 2.5, 300))
    D = np.stack([X - w_ls[0], Y - w_ls[1]], -1)
    Q = np.einsum("...i,ij,...j->...", D, H, D)
    for ax, kind in zip(axes[:2], ["l1", "l2"]):
        if kind == "l1":
            ball = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
            ax.fill(*ball.T, color=RED, alpha=0.2)
            ax.plot(*ball.T, color=RED, linewidth=1)
            sol = np.array([1.0, 0.0])
            level = np.einsum("i,ij,j", sol - w_ls, H, sol - w_ls)
            ax.set_title(r"lasso: $\|\mathbf{w}\|_1 \leq t$")
        else:
            t = np.linspace(0, 2 * np.pi, 200)
            ax.fill(np.cos(t), np.sin(t), color=ORANGE, alpha=0.2)
            ax.plot(np.cos(t), np.sin(t), color=ORANGE, linewidth=1)
            ths = np.linspace(0, 2 * np.pi, 4000)
            pts = np.c_[np.cos(ths), np.sin(ths)]
            vals = np.einsum("ni,ij,nj->n", pts - w_ls, H, pts - w_ls)
            sol = pts[np.argmin(vals)]
            level = vals.min()
            ax.set_title(r"ridge: $\|\mathbf{w}\|_2 \leq t$")
        ax.contour(X, Y, Q, levels=[level / 4, level, 2.2 * level], colors=[GRAY, ACCENT, GRAY], linewidths=[0.6, 1.1, 0.6])
        ax.plot(*w_ls, "o", color=ACCENT, markersize=3)
        ax.plot(*sol, "o", color="black", markersize=4)
        ax.axhline(0, color="#dddddd", linewidth=0.6, zorder=0)
        ax.axvline(0, color="#dddddd", linewidth=0.6, zorder=0)
        ax.set_aspect("equal")
        ax.set_xlim(-1.5, 3.2)
        ax.set_ylim(-1.5, 2.5)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].text(2.1, 1.05, r"$\hat{\mathbf{w}}_{\mathrm{LS}}$", color=ACCENT, fontsize=8)
    ax = axes[2]
    v = np.linspace(-3, 3, 400)
    tau = 1.0
    ax.plot(v, v, color="#cccccc", linewidth=0.8, label="identity")
    ax.plot(v, np.sign(v) * np.maximum(np.abs(v) - tau, 0), color=RED, linewidth=1.4, label=r"soft threshold $S_1(v)$")
    ax.plot(v, v / (1 + 2 * 0.5), color=ORANGE, linewidth=1.2, linestyle="--", label=r"ridge: $v / (1 + 2\lambda)$")
    ax.set_title("prox steps")
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", handlelength=1.4)
    ax.set_xlabel("$v$", labelpad=1)
    ax.tick_params(labelsize=7)
    save(fig, "s08_lasso")


@figure("s08")
def s08_svm():
    """A maximum-margin separating hyperplane x1 + x2 = 2, its margins and the support vectors."""
    pos = np.array([[1.5, 1.5], [2.5, 0.5], [3.0, 1.5], [2.2, 2.6], [3.3, 2.8], [1.4, 3.1], [3.6, 0.8]])
    neg = np.array([[0.5, 0.5], [-0.3, 1.3], [0.2, -0.4], [-0.8, 0.4], [-0.5, -0.9], [0.9, -1.0], [-1.2, 1.6]])
    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    x = np.linspace(-1.6, 4, 2)
    ax.fill_between(x, 1 - x, 3 - x, color=ACCENT, alpha=0.08)
    ax.plot(x, 2 - x, color="black", linewidth=1.2)
    ax.plot(x, 3 - x, color=GRAY, linewidth=0.8, linestyle="--")
    ax.plot(x, 1 - x, color=GRAY, linewidth=0.8, linestyle="--")
    ax.scatter(*pos.T, color=RED, s=14, zorder=3)
    ax.scatter(*neg.T, color=ACCENT, s=14, marker="s", zorder=3)
    sv = np.array([[1.5, 1.5], [2.5, 0.5], [0.5, 0.5]])
    ax.scatter(*sv.T, s=60, facecolors="none", edgecolors="black", linewidths=0.9, zorder=4)
    ax.annotate("", xy=(1.5, 1.5), xytext=(1.0, 1.0), arrowprops=dict(arrowstyle="<->", color=GREEN, linewidth=0.9))
    ax.text(1.32, 0.93, r"$\frac{1}{\|\mathbf{w}\|}$", color=GREEN, fontsize=9)
    ax.text(3.0, -1.3, r"$\mathbf{w}^\top\mathbf{x} + b = 0$", fontsize=7.5)
    ax.set_xlim(-1.6, 4)
    ax.set_ylim(-1.6, 3.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("circled: support vectors ($\\mu_i > 0$)", fontsize=8)
    save(fig, "s08_svm")


# ---------------------------------------------------------------- Lecture 9: probability I-II

@figure("s09")
def s09_bayes():
    """Natural frequencies for the medical test: 2000 people, prevalence 1%, sensitivity 99%, false positive rate 5%."""
    rows, cols = 40, 50
    n = rows * cols
    sick = 20
    true_pos = 20  # 99% of 20, rounded
    false_pos = 99  # 5% of 1980
    kinds = np.zeros(n, dtype=int)  # 0 negative, 1 healthy but positive, 2 sick and positive
    kinds[:true_pos] = 2
    kinds[true_pos:true_pos + false_pos] = 1
    colors = np.array(["#dddddd", ORANGE, RED])
    yy, xx = np.divmod(np.arange(n), cols)
    fig, ax = plt.subplots(figsize=(3.3, 2.75))
    ax.scatter(xx, -yy, s=4.5, c=colors[kinds], marker="s", linewidths=0)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(r"2000 people, positive tests: $\approx 20$ sick (red), $99$ healthy (orange)", fontsize=7.5)
    save(fig, "s09_bayes")


@figure("s09")
def s09_inverse_transform():
    """Inverse transform sampling for Exponential(1): u -> F^{-1}(u) = -log(1 - u)."""
    rng = np.random.default_rng(2)
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.95))
    ax = axes[0]
    x = np.linspace(0, 4, 300)
    ax.plot(x, 1 - np.exp(-x), color=ACCENT, linewidth=1.3)
    for u in [0.2, 0.5, 0.8, 0.95]:
        xu = -np.log(1 - u)
        ax.annotate("", xy=(xu, u), xytext=(0, u), arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=0.8, linestyle="--"))
        ax.annotate("", xy=(xu, 0), xytext=(xu, u), arrowprops=dict(arrowstyle="-|>", color=ORANGE, linewidth=0.8))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("$x$", labelpad=1)
    ax.set_title(r"CDF $F(x) = 1 - e^{-x}$: $u \mapsto F^{-1}(u)$")
    ax.tick_params(labelsize=7)
    ax = axes[1]
    s = -np.log(1 - rng.uniform(size=5000))
    ax.hist(s, bins=40, range=(0, 5), density=True, color=ACCENT, alpha=0.35)
    xx = np.linspace(0, 5, 200)
    ax.plot(xx, np.exp(-xx), color=RED, linewidth=1.2, label="density $e^{-x}$")
    ax.set_title(r"histogram of $5000$ samples $-\log(1 - U)$")
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(labelsize=7)
    save(fig, "s09_inverse_transform")


@figure("s09")
def s09_correlation():
    """Scatter plots with different correlations, and a dependent but uncorrelated pair."""
    rng = np.random.default_rng(5)
    n = 300
    fig, axes = plt.subplots(1, 5, figsize=(W, 1.25))
    for ax, rho in zip(axes[:4], [-0.9, 0.0, 0.5, 0.95]):
        X = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
        ax.scatter(*X.T, s=1.5, color=ACCENT)
        ax.set_title(f"$\\rho = {rho}$", fontsize=8)
    x = rng.uniform(-1.7, 1.7, n)
    y = x ** 2 + rng.normal(0, 0.1, n) - 1
    ax = axes[4]
    ax.scatter(x, y, s=1.5, color=RED)
    ax.set_title(r"$Y = X^2$: $\rho = 0$", fontsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#cccccc")
    save(fig, "s09_correlation")


@figure("s09")
def s09_initialization():
    """Second moment of the activations through 50 ReLU layers of width 512 for three weight variances."""
    rng = np.random.default_rng(6)
    n, L = 512, 50
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    for v, color, label in [(1.0, RED, r"$v = 1$ (standard normal)"), (2.0 / n, GREEN, r"$v = 2/n$ (He)"), (1.0 / n, ACCENT, r"$v = 1/n$")]:
        a = rng.normal(0, 1, (n, 64))
        q = [np.mean(a ** 2)]
        for _ in range(L):
            W_ = rng.normal(0, np.sqrt(v), (n, n))
            a = np.maximum(W_ @ a, 0)
            q.append(np.mean(a ** 2))
        ax.semilogy(np.arange(L + 1), q, color=color, label=label)
    ax.set_xlabel("layer $l$", labelpad=1)
    ax.set_ylabel(r"$q_l = \mathbb{E}[a^2]$")
    ax.set_ylim(1e-17, 1e120)
    ax.set_yticks([1e-15, 1, 1e15, 1e60, 1e100])
    ax.legend(frameon=False, fontsize=7, loc="center left")
    ax.tick_params(labelsize=7)
    save(fig, "s09_initialization")


@figure("s09")
def s09_gaussian_conditional():
    """A 2D Gaussian with correlation 0.5: the conditional distribution of Y given X = 2 is N(1, 0.75)."""
    rho = 0.5
    S = np.array([[1, rho], [rho, 1]])
    Si = np.linalg.inv(S)
    x = np.linspace(-3.2, 3.2, 300)
    X, Y = np.meshgrid(x, x)
    Z = np.exp(-0.5 * (Si[0, 0] * X ** 2 + 2 * Si[0, 1] * X * Y + Si[1, 1] * Y ** 2))
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.2), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    ax.contour(X, Y, Z, levels=[0.05, 0.2, 0.45, 0.75], colors=ACCENT, linewidths=0.8)
    ax.plot(x, rho * x, color=RED, linewidth=1.1, label=r"$\mathbb{E}[Y \mid X = x] = \rho x$")
    ax.plot(x, x, color=GRAY, linewidth=0.8, linestyle=":", label="major axis $y = x$")
    ax.axvline(2, color=ORANGE, linewidth=1, linestyle="--")
    ax.set_aspect("equal")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel("$x$", labelpad=1)
    ax.set_ylabel("$y$", labelpad=1)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.tick_params(labelsize=7)
    ax = axes[1]
    y = np.linspace(-2.5, 4, 300)
    ax.plot(y, np.exp(-0.5 * y ** 2) / np.sqrt(2 * np.pi), color=GRAY, label=r"marginal $\mathcal{N}(0, 1)$")
    ax.plot(y, np.exp(-0.5 * (y - 1) ** 2 / 0.75) / np.sqrt(2 * np.pi * 0.75), color=ORANGE, linewidth=1.3,
            label=r"given $X = 2$: $\mathcal{N}(1, 0.75)$")
    ax.set_xlabel("$y$", labelpad=1)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_ylim(0, 0.62)
    ax.tick_params(labelsize=7)
    save(fig, "s09_gaussian_conditional")


# ---------------------------------------------------------------- Lecture 10: probability III

@figure("s10")
def s10_lln_clt():
    """LLN: running means of coin flips inside the band 1/(2 sqrt N); CLT: standardized sums of exponentials."""
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.1))
    ax = axes[0]
    N = np.arange(1, 1001)
    for k in range(6):
        flips = rng.integers(0, 2, 1000)
        ax.plot(N, np.cumsum(flips) / N, linewidth=0.7, alpha=0.85)
    ax.fill_between(N, 0.5 - 1 / np.sqrt(N), 0.5 + 1 / np.sqrt(N), color=GRAY, alpha=0.15, label=r"$\frac{1}{2} \pm 2\frac{\sigma}{\sqrt{N}}$")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("$N$", labelpad=1)
    ax.set_title(r"running means $\bar X_N$ of coin flips")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.tick_params(labelsize=7)
    ax = axes[1]
    z = np.linspace(-3.5, 3.5, 300)
    for n, c in [(1, ORANGE), (4, GREEN), (30, ACCENT)]:
        S = rng.exponential(1, (40000, n)).sum(1)
        Zs = (S - n) / np.sqrt(n)
        hist, edges = np.histogram(Zs, bins=70, range=(-3.5, 3.5), density=True)
        ax.step(edges[:-1], hist, where="post", color=c, linewidth=0.9, label=f"$N = {n}$")
    ax.plot(z, np.exp(-z ** 2 / 2) / np.sqrt(2 * np.pi), color="black", linewidth=1.2, linestyle="--", label=r"$\varphi(z)$")
    ax.set_title(r"$\sqrt{N}(\bar X_N - 1)$, $X_i \sim$ Exponential$(1)$")
    ax.legend(frameon=False, fontsize=7)
    ax.set_ylim(0, 0.95)
    ax.tick_params(labelsize=7)
    save(fig, "s10_lln_clt")


@figure("s10")
def s10_hoeffding():
    """P(|mean of N fair coins - 1/2| >= 0.1): exact, CLT, Hoeffding and Chebyshev."""
    from math import comb, erf, sqrt
    Ns = np.arange(10, 401, 5)
    eps = 0.1
    exact = []
    for n in Ns:
        tot = 0.0
        for k in range(n + 1):
            if abs(k / n - 0.5) >= eps - 1e-12:
                tot += comb(n, k)
        exact.append(tot / 2.0 ** n)
    clt = [1 - erf(eps * 2 * sqrt(n) / sqrt(2)) for n in Ns]
    hoeff = np.minimum(1, 2 * np.exp(-2 * Ns * eps ** 2))
    cheb = np.minimum(1, 0.25 / (Ns * eps ** 2))
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ax.semilogy(Ns, cheb, color=ORANGE, label="Chebyshev $\\frac{1}{4N\\varepsilon^2}$")
    ax.semilogy(Ns, hoeff, color=RED, label="Hoeffding $2e^{-2N\\varepsilon^2}$")
    ax.semilogy(Ns, clt, color=GREEN, linestyle="--", label="CLT approximation")
    ax.semilogy(Ns, exact, color=ACCENT, linewidth=1.2, label="exact")
    ax.set_xlabel("$N$", labelpad=1)
    ax.set_title(r"$P(|\bar X_N - \frac{1}{2}| \geq 0.1)$ for fair coins")
    ax.set_ylim(1e-6, 2)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.tick_params(labelsize=7)
    save(fig, "s10_hoeffding")


@figure("s10")
def s10_monte_carlo():
    """Monte Carlo estimate of pi from points in the square, and its error ~ 1/sqrt(N)."""
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.1), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    P = rng.uniform(-1, 1, (1500, 2))
    inside = (P ** 2).sum(1) <= 1
    ax.scatter(*P[inside].T, s=1.5, color=ACCENT)
    ax.scatter(*P[~inside].T, s=1.5, color=ORANGE)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color="black", linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(r"$N = 1500$: $\pi \approx 4 \cdot$ (fraction inside) $= %.3f$" % (4 * inside.mean()), fontsize=7.5)
    ax = axes[1]
    Ns = np.logspace(1, 6, 30).astype(int)
    errs = []
    for n in Ns:
        e = [abs(4 * ((rng.uniform(-1, 1, (n, 2)) ** 2).sum(1) <= 1).mean() - np.pi) for _ in range(20)]
        errs.append(np.sqrt(np.mean(np.square(e))))
    ax.loglog(Ns, errs, "o", color=ACCENT, markersize=3, label="RMS error (20 runs)")
    sd = np.sqrt(np.pi / 4 * (1 - np.pi / 4)) * 4
    ax.loglog(Ns, sd / np.sqrt(Ns), color=RED, linewidth=1, label=r"$\sigma / \sqrt{N}$")
    ax.set_xlabel("$N$", labelpad=1)
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("error of the Monte Carlo estimate")
    ax.tick_params(labelsize=7)
    save(fig, "s10_monte_carlo")


@figure("s10")
def s10_metropolis():
    """Random-walk Metropolis for a bimodal density: trace and histogram."""
    rng = np.random.default_rng(9)

    def logp(x):
        return np.logaddexp(np.log(0.3) - 0.5 * ((x + 2) / 0.6) ** 2 - np.log(0.6),
                            np.log(0.7) - 0.5 * ((x - 1.5) / 0.9) ** 2 - np.log(0.9))

    T = 20000
    x = 0.0
    xs = np.empty(T)
    for t in range(T):
        y = x + rng.normal(0, 1.2)
        if np.log(rng.uniform()) < logp(y) - logp(x):
            x = y
        xs[t] = x
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.9), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    ax.plot(xs[:600], color=ACCENT, linewidth=0.6)
    ax.set_xlabel("step $t$", labelpad=1)
    ax.set_title("trace of the chain (first 600 steps)")
    ax.tick_params(labelsize=7)
    ax = axes[1]
    grid = np.linspace(-4.5, 5, 300)
    dens = np.exp(logp(grid)) / np.sqrt(2 * np.pi)
    ax.hist(xs[1000:], bins=60, range=(-4.5, 5), density=True, color=ACCENT, alpha=0.35, label="samples")
    ax.plot(grid, dens, color=RED, linewidth=1.2, label="target $p$")
    ax.set_title(f"histogram of {T - 1000} samples")
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(labelsize=7)
    save(fig, "s10_metropolis")


@figure("s10")
def s10_gridworld():
    """Value iteration on a 4x5 gridworld: optimal values and the greedy policy."""
    H, Wd = 4, 5
    goal, trap, wall = (0, 4), (1, 4), (1, 1)
    gamma = 0.9
    moves = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    states = [(i, j) for i in range(H) for j in range(Wd) if (i, j) != wall]

    def step(s, a):
        d = moves[a]
        t = (s[0] + d[0], s[1] + d[1])
        if not (0 <= t[0] < H and 0 <= t[1] < Wd) or t == wall:
            return s
        return t

    def reward(s):
        return 1.0 if s == goal else (-1.0 if s == trap else -0.04)

    V = {s: 0.0 for s in states}
    for _ in range(200):
        newV = {}
        for s in states:
            if s in (goal, trap):
                newV[s] = reward(s)
                continue
            best = -np.inf
            for a in moves:
                # the intended move succeeds with probability 0.8, otherwise a perpendicular one
                perp = ["left", "right"] if a in ("up", "down") else ["up", "down"]
                val = 0.8 * V[step(s, a)] + 0.1 * V[step(s, perp[0])] + 0.1 * V[step(s, perp[1])]
                best = max(best, val)
            newV[s] = reward(s) + gamma * best
        V = newV
    fig, ax = plt.subplots(figsize=(3.0, 2.4))
    grid = np.full((H, Wd), np.nan)
    for s in states:
        grid[s] = V[s]
    ax.imshow(grid, cmap="RdYlGn", vmin=-1, vmax=1)
    arrows = {"up": (0, -0.3), "down": (0, 0.3), "left": (-0.3, 0), "right": (0.3, 0)}
    for s in states:
        i, j = s
        ax.text(j, i + 0.28, f"{V[s]:.2f}", ha="center", va="center", fontsize=6.5)
        if s in (goal, trap):
            ax.text(j, i - 0.1, "+1" if s == goal else "-1", ha="center", va="center", fontsize=9, fontweight="bold")
            continue
        best_a, best_v = None, -np.inf
        for a in moves:
            perp = ["left", "right"] if a in ("up", "down") else ["up", "down"]
            val = 0.8 * V[step(s, a)] + 0.1 * V[step(s, perp[0])] + 0.1 * V[step(s, perp[1])]
            if val > best_v:
                best_a, best_v = a, val
        dx, dy = arrows[best_a]
        ax.annotate("", xy=(j + dx, i - 0.1 + dy), xytext=(j - dx, i - 0.1 - dy),
                    arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.0))
    ax.add_patch(plt.Rectangle((wall[1] - 0.5, wall[0] - 0.5), 1, 1, color="#555555"))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(r"$V^*$ and the greedy policy ($\gamma = 0.9$)", fontsize=8)
    save(fig, "s10_gridworld")


@figure("s10")
def s10_kl():
    """Fitting a Gaussian q to a bimodal p: KL(p||q) covers the mass, KL(q||p) picks a mode."""
    x = np.linspace(-6, 6, 2401)
    dx = x[1] - x[0]

    def normal(x, m, s):
        return np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))

    p = 0.5 * normal(x, -2, 0.7) + 0.5 * normal(x, 2, 0.7)
    mean = (x * p).sum() * dx
    var = ((x - mean) ** 2 * p).sum() * dx
    q_fwd = normal(x, mean, np.sqrt(var))
    best, best_ms = np.inf, None
    for m in np.linspace(-3, 3, 121):
        for s in np.linspace(0.3, 3, 55):
            q = normal(x, m, s)
            kl = (q * (np.log(q + 1e-300) - np.log(p + 1e-300))).sum() * dx
            if kl < best:
                best, best_ms = kl, (m, s)
    q_rev = normal(x, *best_ms)
    fig, ax = plt.subplots(figsize=(3.6, 1.9))
    ax.fill_between(x, p, color=GRAY, alpha=0.3, label="$p$ (data)")
    ax.plot(x, q_fwd, color=ACCENT, linewidth=1.3, label=r"$\min_q \mathrm{KL}(p\,\|\,q)$: covers")
    ax.plot(x, q_rev, color=RED, linewidth=1.3, label=r"$\min_q \mathrm{KL}(q\,\|\,p)$: one mode")
    ax.set_xlim(-5, 5)
    ax.set_ylim(0, 0.62)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.tick_params(labelsize=7)
    save(fig, "s10_kl")


# ---------------------------------------------------------------- Lecture 11: statistical inference

@figure("s11")
def s11_bias_variance():
    """MSE of the shrinkage estimators c * mean for mu = 2, sigma^2 = 16, N = 4."""
    mu, var, N = 2.0, 16.0, 4
    c = np.linspace(0, 1, 200)
    bias2 = (1 - c) ** 2 * mu ** 2
    variance = c ** 2 * var / N
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    ax.plot(c, bias2, color=ORANGE, label=r"bias$^2 = (1 - c)^2\mu^2$")
    ax.plot(c, variance, color=GREEN, label=r"variance $= c^2\sigma^2/N$")
    ax.plot(c, bias2 + variance, color=ACCENT, linewidth=1.5, label="MSE")
    cs = mu ** 2 / (mu ** 2 + var / N)
    ax.plot([cs], [(1 - cs) ** 2 * mu ** 2 + cs ** 2 * var / N], "o", color=RED, markersize=4)
    ax.plot([1], [var / N], "o", color=ACCENT, markersize=4)
    ax.text(1.0, var / N + 0.25, "sample mean", ha="right", fontsize=7, color=ACCENT)
    ax.text(cs, 1.45, "$c^* = 1/2$", ha="center", fontsize=7, color=RED)
    ax.set_xlabel(r"shrinkage factor $c$ in $\hat\mu_c = c\,\bar X$", labelpad=1)
    ax.set_ylim(0, 4.5)
    ax.legend(frameon=False, fontsize=6.5, loc="upper center")
    ax.tick_params(labelsize=7)
    save(fig, "s11_bias_variance")


@figure("s11")
def s11_likelihood():
    """Bernoulli log-likelihoods (minus their maximum) for 7/10 and 70/100: more data, sharper curvature."""
    th = np.linspace(0.3, 0.95, 400)
    fig, ax = plt.subplots(figsize=(3.0, 2.0))
    for k, N, color in [(7, 10, ORANGE), (70, 100, ACCENT)]:
        ll = k * np.log(th) + (N - k) * np.log(1 - th)
        ll -= ll.max()
        ax.plot(th, ll, color=color, label=f"$k = {k}$, $N = {N}$")
    ax.axvline(0.7, color=GRAY, linewidth=0.6, linestyle=":")
    ax.set_ylim(-6, 0.3)
    ax.set_xlabel(r"$\theta$", labelpad=1)
    ax.set_title(r"$\ell(\theta) - \ell(\hat\theta)$; curvature $\approx N\,I(\hat\theta)$", fontsize=8)
    ax.legend(frameon=False, fontsize=7, loc="lower center")
    ax.tick_params(labelsize=7)
    save(fig, "s11_likelihood")


@figure("s11")
def s11_ci_coverage():
    """50 simulated 95% confidence intervals for a mean; the ones that miss are red."""
    rng = np.random.default_rng(12)
    mu, sigma, N = 5.0, 1.5, 25
    fig, ax = plt.subplots(figsize=(W, 1.7))
    misses = 0
    for i in range(50):
        x = rng.normal(mu, sigma, N)
        m, s = x.mean(), x.std(ddof=1)
        lo, hi = m - 1.96 * s / np.sqrt(N), m + 1.96 * s / np.sqrt(N)
        miss = not (lo <= mu <= hi)
        misses += miss
        ax.plot([i, i], [lo, hi], color=RED if miss else ACCENT, linewidth=1.4 if miss else 1.0)
        ax.plot([i], [m], "o", color=RED if miss else ACCENT, markersize=2)
    ax.axhline(mu, color="black", linewidth=0.8)
    ax.text(50.5, mu, r"$\mu$", va="center", fontsize=9)
    ax.set_xticks([])
    ax.set_title(f"50 experiments, intervals $\\bar x \\pm 1.96\\,s/\\sqrt{{N}}$: {misses} miss $\\mu$", fontsize=8)
    ax.tick_params(labelsize=7)
    save(fig, "s11_ci_coverage")


@figure("s11")
def s11_power():
    """The one-sided coin test with 100 flips: rejection region S >= 59, level 0.044, power 0.62 for theta = 0.6."""
    from math import comb
    k = np.arange(30, 81)
    p0 = np.array([comb(100, int(j)) * 0.5 ** 100 for j in k])
    p1 = np.array([comb(100, int(j)) * 0.6 ** j * 0.4 ** (100 - j) for j in k])
    rej = k >= 59
    fig, ax = plt.subplots(figsize=(3.6, 1.9))
    ax.bar(k[~rej], p0[~rej], width=0.9, color=ACCENT, alpha=0.5, label=r"$H_0$: $\theta = 0.5$")
    ax.bar(k[rej], p0[rej], width=0.9, color=RED, alpha=0.9, label=rf"level $\alpha = {p0[rej].sum():.3f}$")
    ax.step(k, p1, where="mid", color=ORANGE, linewidth=1.2, label=rf"$\theta = 0.6$: power $= {p1[rej].sum():.2f}$")
    ax.axvline(58.5, color="black", linewidth=0.8, linestyle="--")
    ax.text(66, 0.07, "reject $H_0$\nif $S \\geq 59$", fontsize=7)
    ax.set_xlabel("number of heads $S$ in 100 flips", labelpad=1)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    ax.set_ylim(0, 0.095)
    ax.tick_params(labelsize=7)
    save(fig, "s11_power")


@figure("s11")
def s11_em():
    """EM for a two-component Gaussian mixture in 2D: the fitted ellipses after 1, 5 and 30 iterations."""
    rng = np.random.default_rng(13)
    A = rng.multivariate_normal([-1.5, 0.5], [[1.0, 0.6], [0.6, 0.8]], 150)
    B = rng.multivariate_normal([1.8, -0.5], [[0.6, -0.3], [-0.3, 1.2]], 150)
    X = np.vstack([A, B])
    K = 2
    pis = np.array([0.5, 0.5])
    mus = np.array([[0.0, 2.0], [0.5, 1.5]])
    Sigmas = np.array([np.eye(2), np.eye(2)])

    def logpdf(X, m, S):
        d = X - m
        Si = np.linalg.inv(S)
        return -0.5 * np.einsum("ni,ij,nj->n", d, Si, d) - 0.5 * np.log(np.linalg.det(S)) - np.log(2 * np.pi)

    snapshots, lls = {}, []
    for it in range(1, 31):
        logr = np.stack([np.log(pis[k]) + logpdf(X, mus[k], Sigmas[k]) for k in range(K)], 1)
        ll = np.logaddexp(logr[:, 0], logr[:, 1])
        lls.append(ll.sum())
        r = np.exp(logr - ll[:, None])
        Nk = r.sum(0)
        pis = Nk / len(X)
        mus = (r.T @ X) / Nk[:, None]
        Sigmas = np.array([((r[:, k, None] * (X - mus[k])).T @ (X - mus[k])) / Nk[k] for k in range(K)])
        if it in (1, 5, 30):
            snapshots[it] = (mus.copy(), Sigmas.copy(), r[:, 0].copy())
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.6), gridspec_kw={"width_ratios": [1, 1, 1, 1.15], "wspace": 0.35})
    t = np.linspace(0, 2 * np.pi, 100)
    for ax, it in zip(axes[:3], (1, 5, 30)):
        mus_, Sig_, r0 = snapshots[it]
        ax.scatter(*X.T, s=2, c=r0, cmap="coolwarm_r", vmin=0, vmax=1)
        for k, c in enumerate([ACCENT, RED]):
            lam, Q = np.linalg.eigh(Sig_[k])
            ell = mus_[k][:, None] + Q @ np.diag(2 * np.sqrt(lam)) @ np.vstack([np.cos(t), np.sin(t)])
            ax.plot(*ell, color=c, linewidth=1.1)
        ax.set_title(f"iteration {it}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4, 4)
    ax = axes[3]
    ax.plot(np.arange(1, 31), lls, color=ACCENT)
    ax.set_title("log-likelihood", fontsize=8)
    ax.set_xlabel("iteration", labelpad=1)
    ax.tick_params(labelsize=6.5)
    save(fig, "s11_em")


def main():
    OUT.mkdir(exist_ok=True)
    decks = set(sys.argv[1:])
    for name, (deck, func) in FIGURES.items():
        if decks and deck not in decks:
            continue
        func()
        print("wrote", name)


if __name__ == "__main__":
    main()
