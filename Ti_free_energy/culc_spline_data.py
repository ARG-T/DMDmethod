import spline
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    phy_X = np.array([1.453676923, 1.7427, 2.031723077, 2.320746154, 2.609769231, 2.898792308, 3.187815385, 3.476838462, 3.765861538, 4.054884615, 4.343907692, 4.632930769, 4.921953846, 5.5, 5.789023077])
    phy_y = np.array([9.524761538, 3.7443, 0.9108, 0.3881, -0.0188, -0.2481, -0.2645, -0.2272, -0.1293, -0.0597, -0.0311, -0.0139, -0.0032, 0, 0])

    rho_X = np.array([1.82038, 2.0558, 2.29122, 2.52664, 2.76206, 2.99748, 3.2329, 3.46832, 3.70374, 3.93916, 4.17458, 4.41, 4.64542])
    rho_y = np.array([1.98292, 1.7475, -5.8678, -8.3376, -5.8399, -3.1141, -1.7257, -0.4429, -0.1467, -0.2096, -0.1442, 0, 0])

    f_X = np.array([1.794222222, 2.0558, 2.317377778, 2.578955556, 2.840533333, 3.102111111, 3.363688889, 3.625266667, 3.886844444, 4.148422222, 4.41, 4.671577778])
    f_y = np.array([-0.873933651, -0.1485, 1.6845, 2.0113, 1.1444, 0.2862, -0.3459, -0.6258, -0.612, -0.3112, 0, 0])

    U_X = np.array([-65.54363333, -55.1423, -44.74096667, -34.33963333, -23.9383, -13.53696667])
    U_y = np.array([-0.3785904, -0.29746, -0.15449, 0.05099, 0.57343, 1.667650267])

    g_X = np.array([-1.275485714, -1, -0.724514286, -0.449028571, -0.173542857, 0.101942857, 0.377428571, 0.652914286, 0.9284, 1.203885714])
    g_y = np.array([-2.220059109, 0.0765, 0.1416, 0.7579, 0.6301, 0.0905, -0.3574, -0.6529, -6.0091, -22.64912586])

    for X, y in ((phy_X, phy_y), (rho_X, rho_y), (f_X, f_y), (U_X, U_y), (g_X, g_y)):
        cS = spline.cubicSpline(X, y)
        plt.scatter(X[1:-1], y[1:-1])

        x = np.arange(X[1], X[-2], 0.01)
        plt.plot(x, list(map(cS.fit, x)))
        plt.show()