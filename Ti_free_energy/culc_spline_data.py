import spline
import numpy as np


if __name__ == "__main__":
    X = np.array([1.453676923, 1.7427, 2.031723077, 2.320746154, 2.609769231, 2.898792308, 3.187815385, 3.476838462, 3.765861538, 4.054884615, 4.343907692, 4.632930769, 4.921953846, 5.5, 5.789023077])
    y = np.array([9.524761538, 3.7443, 0.9108, 0.3881, -0.0188, -0.2481, -0.2645, -0.2272, -0.1293, -0.0597, -0.0311, -0.0139, -0.0032, 0, 0])
    cS = spline.cubicSpline(X, y)