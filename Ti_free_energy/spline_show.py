import spline
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    X = np.array([1.794222222, 2.0558, 2.317377778, 2.578955556, 2.840533333, 3.102111111, 3.363688889, 3.625266667, 3.886844444, 4.148422222, 4.41, 4.671577778])
    y = np.array([-0.873933651, -0.1485, 1.6845, 2.0113, 1.1444, 0.2862, -0.3459, -0.6258, -0.612, -0.3112, 0, 0])
    cS = spline.cubicSpline(X, y)
    plt.scatter(X, y)

    x = np.arange(1.8, 4.7, 0.01)
    plt.plot(x, list(map(cS.fit, x)))
    plt.show()
