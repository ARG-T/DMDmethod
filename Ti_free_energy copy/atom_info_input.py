import csv
import numpy as np


def pos_input(a=2.956, c=4.708, temparature=300):
    x, y, z = 3, 3, 3
    atom_info = np.array([[0]*6 for _ in range(8*x*y*z)], dtype=float)
    sq_3 = 3**(0.5)
    base = [[0, 0, 0], [a/2, sq_3*a/2, 0], [a, 0, 0], [3*a/2, sq_3*a/2, 0], [0, sq_3*a/3, c/2], [a/2, 5*sq_3*a/6, c/2], [a, sq_3*a/3, c/2], [3*a/2, 5*sq_3*a/6, c/2]]
    row = 0
    for i in range(x):
        for j in range(y):
            for k in range(z):
                for each in range(8):
                    atom_info[row][0] = int(row)+1
                    atom_info[row][1] = base[each][0] + 2*a*i
                    atom_info[row][2] = base[each][1] + sq_3*a*j
                    atom_info[row][3] = base[each][2] + c*k
                    atom_info[row][4] = 1-10**(-8)
                    atom_info[row][5] = 300
                    row += 1

    with open("atom_info.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for info in atom_info:
            writer.writerow(info)
