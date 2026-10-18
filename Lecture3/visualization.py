import numpy as np
import matplotlib.pyplot as plt


def main():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    z = x**2 + y**2

    plt.figure(figsize=(8, 6))
    plt.contour(x, y, z.reshape(100, 100), levels=20)
    plt.title("Contour Plot of $z = x^2 + y^2$")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
