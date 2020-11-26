import csv
import numpy as np


def pos_input(lat_p, temparature):
    x, y, z = 3, 3, 3
    lattice = lat_p
    temparature = temparature
    atom_info = np.array([[0]*6 for _ in range(4*x*y*z)], dtype=float)
    base = [[0, 0, 0], [0, lattice/2, lattice/2], [lattice/2, 0, lattice/2], [lattice/2, lattice/2, 0]]
    for i in range(x):
        for j in range(y):
            for k in range(z):
                base_row = 4*(9*i+3*j+k)
                for each in range(4):
                    row = base_row+each
                    atom_info[row][0] = int(row)+1
                    atom_info[row][1] = base[each][0] + lattice*i
                    atom_info[row][2] = base[each][1] + lattice*j
                    atom_info[row][3] = base[each][2] + lattice*k
                    atom_info[row][4] = 1-10**(-8)
                    atom_info[row][5] = 4.108*10**4/temparature

    with open("atom_info.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for info in atom_info:
            writer.writerow(info)
