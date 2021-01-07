import numpy as np
import csv
import Ti_atom_info
import O_atom_info
import math
from free_energy_culc import Culculation as Cc
from joblib import Parallel, delayed


M_PI = math.pi
PLANCK_CONSTANT = 6.62607015        # 10**-34
BOLTZMANN_CONSTANT = 1.380649e-1     # 10**-22
BOLTZMANN_CONSTANT_EV = 8.6173336e-5


class FreeEnergy():

    def __init__(self, temperature):
        self.Ti_mass = 7.99379      # 10**-26
        self.O_mass = 0.167     # 10**-26
        self.outer_cutoff = 5.5
        self.lat_parameter_a = 2.951
        self.lat_parameter_c = 4.6843
        self.lat_parameter_x = 6*self.lat_parameter_a
        self.lat_parameter_y = 3**(1.5)*self.lat_parameter_a
        self.lat_parameter_z = 3*self.lat_parameter_c
        self.Ti_atom_num = 216
        self.O_atom_num = 1
        self.atom_num = self.Ti_atom_num + self.O_atom_num
        self.Ti_x_pos = np.zeros(self.Ti_atom_num, dtype=float)
        self.Ti_y_pos = np.zeros(self.Ti_atom_num, dtype=float)
        self.Ti_z_pos = np.zeros(self.Ti_atom_num, dtype=float)
        self.O_x_pos = np.zeros(self.O_atom_num, dtype=float)
        self.O_y_pos = np.zeros(self.O_atom_num, dtype=float)
        self.O_z_pos = np.zeros(self.O_atom_num, dtype=float)
        self.atom_x_pos = np.zeros(self.atom_num, dtype=float)
        self.atom_y_pos = np.zeros(self.atom_num, dtype=float)
        self.atom_z_pos = np.zeros(self.atom_num, dtype=float)
        self.Ti_gauss_width = np.zeros(self.Ti_atom_num, dtype=float)
        self.O_gauss_width = np.zeros(self.O_atom_num, dtype=float)
        self.Ti_atom_mass = np.zeros(self.Ti_atom_num, dtype=float)
        self.O_atom_mass = np.zeros(self.O_atom_num, dtype=float)
        self.atom_mass = np.zeros(self.atom_num, dtype=float)
        self.atom_gauss_width = np.zeros(self.atom_num, dtype=float)
        self.current_total_energy = 0
        self.sigma = 1e-10
        self.temperature = temperature
        self.vib_ent_list = np.zeros(self.atom_num, dtype=float)
        self.Ti_occupancy = np.zeros(self.Ti_atom_num, dtype=float)
        self.O_occupancy = np.zeros(self.O_atom_num, dtype=float)
        self.atom_gauss_width = np.zeros(self.atom_num, dtype=float)
        self.energy_list = np.zeros(self.atom_num, dtype=float)
        self.VG_energy_list = np.zeros(self.atom_num, dtype=float)
        self.vib_ent_list = np.zeros(self.atom_num, dtype=float)
        self.mixed_list = np.zeros(self.atom_num, dtype=float)
        self.reference_neighbor_list = []
        self.reference_neighbor_list_x_pos = []
        self.reference_neighbor_list_y_pos = []
        self.reference_neighbor_list_z_pos = []
        self.reference_neighbor_list_occ = []
        self.reference_neighbor_list_alpha = []
        self.Verlet_neighbor_list = np.zeros(self.atom_num+1, dtype=int)

    # ドブロイ波長
    def thermal_wavelength(self):
        ret = PLANCK_CONSTANT/((2*M_PI*self.atom_mass*BOLTZMANN_CONSTANT*self.temperature)**0.5)
        return ret

    # 初期配置入力
    def pos_init(self):
        with open("Ti_atom_info.csv") as f:
            reader = csv.reader(f)
            info = [row for row in reader]

        for i in range(self.Ti_atom_num):
            self.Ti_x_pos[i] = float(info[i][1])
            self.Ti_y_pos[i] = float(info[i][2])
            self.Ti_z_pos[i] = float(info[i][3])
            self.Ti_occupancy[i] = float(info[i][4])
            self.Ti_gauss_width[i] = float(info[i][5])
            self.Ti_atom_mass[i] = self.Ti_mass

        with open("O_atom_info.csv") as f:
            reader = csv.reader(f)
            info = [row for row in reader]

        for i in range(self.O_atom_num):
            self.O_x_pos[i] = float(info[i][1])
            self.O_y_pos[i] = float(info[i][2])
            self.O_z_pos[i] = float(info[i][3])
            self.O_occupancy[i] = float(info[i][4])
            self.O_gauss_width[i] = float(info[i][5])
            self.O_atom_mass[i] = self.O_mass

        self.atom_x_pos = np.concatenate([self.Ti_x_pos, self.O_x_pos])
        self.atom_y_pos = np.concatenate([self.Ti_y_pos, self.O_y_pos])
        self.atom_z_pos = np.concatenate([self.Ti_z_pos, self.O_z_pos])
        self.atom_occupancy = np.concatenate([self.Ti_occupancy, self.O_occupancy])
        self.atom_gauss_width = np.concatenate([self.Ti_gauss_width, self.O_gauss_width])
        self.atom_mass = np.concatenate([self.Ti_atom_mass, self.O_atom_mass])

    # 同等の位置を考慮しつつ二つの原子の距離(A)を返す
    def abs_two_atom_distance(self, base, pair):
        base_x, base_y, base_z = self.atom_x_pos[base], self.atom_y_pos[base], self.atom_z_pos[base]
        pair_x, pair_y, pair_z = self.atom_x_pos[pair], self.atom_y_pos[pair], self.atom_z_pos[pair]
        dx = abs(pair_x - base_x)
        dx = min(dx, self.lat_parameter_x-dx)
        dy = abs(pair_y - base_y)
        dy = min(dy, self.lat_parameter_y-dy)
        dz = abs(pair_z - base_z)
        dz = min(dz, self.lat_parameter_z-dz)
        dr2 = dx**2 + dy**2 + dz**2
        return dr2

    def alpha_int(self, i, j):
        o_i = self.atom_gauss_width[i]
        o_j = self.atom_gauss_width[j]
        return o_i*o_j/(o_i+o_j)

    # 近接リスト作成
    def make_Verlet_neighbor_list(self):
        nlist = -1      # 0-indexed
        for i in range(self.atom_num):
            self.Verlet_neighbor_list[i] = nlist+1
            for j in range(self.atom_num):
                if not i == j:
                    dr2 = self.abs_two_atom_distance(i, j)
                    if dr2 <= self.outer_cutoff**2:
                        nlist += 1
                        self.reference_neighbor_list.append(j)  # 原子number
                        self.reference_neighbor_list_x_pos.append(self.atom_x_pos[j])   # x_pos
                        self.reference_neighbor_list_y_pos.append(self.atom_y_pos[j])   # y_pos
                        self.reference_neighbor_list_z_pos.append(self.atom_z_pos[j])   # z_pos
                        self.reference_neighbor_list_occ.append(self.atom_occupancy[j])   # occupancy
                        self.reference_neighbor_list_alpha.append(self.alpha_int(i, j))   # alpha
        self.Verlet_neighbor_list[-1] = nlist+1
        self.reference_neighbor_list = np.array(self.reference_neighbor_list, dtype=int)
        self.reference_neighbor_list_x_pos = np.array(self.reference_neighbor_list_x_pos, dtype=float)
        self.reference_neighbor_list_y_pos = np.array(self.reference_neighbor_list_y_pos, dtype=float)
        self.reference_neighbor_list_z_pos = np.array(self.reference_neighbor_list_z_pos, dtype=float)
        self.reference_neighbor_list_occ = np.array(self.reference_neighbor_list_occ, dtype=float)
        self.reference_neighbor_list_alpha = np.array(self.reference_neighbor_list_alpha, dtype=float)

    def culc_VG_energy(self, i):
        start = self.Verlet_neighbor_list[i]
        end = self.Verlet_neighbor_list[i+1]
        if start > end:
            return 0
        i_pos_x = self.atom_x_pos[i]
        i_pos_y = self.atom_y_pos[i]
        i_pos_z = self.atom_z_pos[i]
        j_list = self.reference_neighbor_list[start:end]
        j_list_x = self.reference_neighbor_list_x_pos[start:end]
        j_list_y = self.reference_neighbor_list_y_pos[start:end]
        j_list_z = self.reference_neighbor_list_z_pos[start:end]
        j_list_occ = self.reference_neighbor_list_occ[start:end]
        j_list_alpha = self.reference_neighbor_list_alpha[start:end]
        ret_data = Cc(i, i_pos_x, i_pos_y, i_pos_z, j_list, j_list_x, j_list_y, j_list_z, j_list_occ, j_list_alpha)
        pair_i, embed_i = ret_data.culc_data()
        ret = pair_i + embed_i
        return ret

    def mixed_entropy(self):
        occ = self.atom_occupancy
        return occ*np.log(occ) + (1-occ)*np.log(1-occ)

    def culc_all_total_energy(self):
        # pair, embedのリストと総和
        print(self.culc_VG_energy(216))
        ret_list = Parallel(n_jobs=-1)([delayed(self.culc_VG_energy)(i) for i in range(self.atom_num)])
        self.VG_energy_list = self.atom_occupancy*np.array(ret_list)
        VG_energy = np.sum(self.VG_energy_list)
        # vib_ent_listの作成と総和
        self.vib_ent_list = np.log((self.atom_gauss_width*self.thermal_wavelength()**2)/M_PI)-1
        vib_energy = 1.5*BOLTZMANN_CONSTANT_EV*self.temperature*np.sum(self.atom_occupancy*self.vib_ent_list)
        # mixed_ent_listの作成と総和
        self.mixed_list = self.mixed_entropy()
        mixed_energy = BOLTZMANN_CONSTANT_EV*self.temperature*np.sum(self.mixed_list)
        print(vib_energy, mixed_energy)
        return VG_energy + vib_energy + mixed_energy

    def main_loop(self):
        Ti_atom_info.pos_input()
        O_atom_info.pos_input()
        # 初期入力
        self.pos_init()
        self.make_Verlet_neighbor_list()
        # print(self.Verlet_neighbor_list)
        # while start
        # while True:
        # 自由エネルギー算出
        self.current_total_energy = self.culc_all_total_energy()
        print("free_energy is ", self.current_total_energy)
        # エネルギーの最小化
        # self.update_info()
        # 濃度時間変化
        # self.update_concentration()
        # while
