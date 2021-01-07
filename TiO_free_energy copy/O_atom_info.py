import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def pos_input(a=2.951, c=4.6843, temparature=300):
    atom_info = np.array([[0]*6], dtype=float)
    sq_3 = 3**(1/2)
    row = 0
    atom_info[row][0] = int(row)+1
    atom_info[row][1] = a/2
    atom_info[row][2] = sq_3*a/6
    atom_info[row][3] = 3*c/4
    atom_info[row][4] = 1-10**(-8)
    atom_info[row][5] = 150

    """
    fig = plt.figure()
    ax = Axes3D(fig)
    X = atom_info[0][1]
    Y = atom_info[0][2]
    Z = atom_info[0][3]
    ax.scatter(X, Y, Z, c="blue")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.plot(X, Y, Z, marker="o", linestyle="None")
    plt.xlim([0, 18])
    plt.ylim([0, 18])
    plt.show()
    """

    with open("O_atom_info.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for info in atom_info:
            writer.writerow(info)
