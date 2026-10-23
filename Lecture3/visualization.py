import numpy as np
import matplotlib.pyplot as plt


def main():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)  # 100 x 100 grids of the x- and y-coordinates
    Z = X**2 + Y**2

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=20)
    plt.gca().set_aspect("equal")  # so that the circles look like circles
    plt.title("Contour Plot of $z = x^2 + y^2$")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
