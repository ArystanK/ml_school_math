import numpy as np
import matplotlib.pyplot as plt
import math


def main():
    probability = 0
    for i in range(50, 101):
        probability += math.comb(100, i)
    print(probability / (2 ** 100))
    print(1 / (2 ** 100))


if __name__ == "__main__":
    main()
