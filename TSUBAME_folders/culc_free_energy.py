#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
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
        self.lat_parameter_x = 29.51
        self.lat_parameter_y = 69.3
        self.lat_parameter_z = 91
        self.Ti_atom_num = 10400
        self.O_atom_num = 7
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
        self.atom_gauss_width = np.zeros(self.atom_num, dtype=float)
        self.current_total_energy = 0
        self.gauss_sigma = 1
        self.pos_sigma = 1e-3
        self.rate = 0.1
        self.delta_Q = 2.04181      # eV
        self.diff_coef = 4.08*10**15        # ang^2/s
        self.time_step = 10**(-4)
        self.temperature = temperature
        self.vib_ent_list = np.zeros(self.atom_num, dtype=float)
        self.Ti_occupancy = np.zeros(self.Ti_atom_num, dtype=float)
        self.O_occupancy = np.zeros(self.O_atom_num, dtype=float)
        self.VG_energy_list = np.zeros(self.O_atom_num, dtype=float)
        self.pair_and_embed = np.zeros(self.O_atom_num, dtype=float)
        self.mixed_list = np.zeros(self.O_atom_num, dtype=float)
        self.form_energy = np.zeros(self.O_atom_num, dtype=float)
        self.delta_occupancy = np.zeros(self.O_atom_num, dtype=float)
        self.reference_neighbor_list = []
        self.reference_neighbor_list_x_pos = []
        self.reference_neighbor_list_y_pos = []
        self.reference_neighbor_list_z_pos = []
        self.reference_neighbor_list_occ = []
        self.reference_neighbor_list_alpha = []
        self.Verlet_neighbor_list = np.zeros(self.O_atom_num+1, dtype=int)
        self.gauss_differential_list = np.zeros(self.O_atom_num, dtype=float)
        self.x_differential_list = np.zeros(self.O_atom_num, dtype=float)
        self.y_differential_list = np.zeros(self.O_atom_num, dtype=float)
        self.z_differential_list = np.zeros(self.O_atom_num, dtype=float)

    # ドブロイ波長
    def thermal_wavelength(self):
        ret = PLANCK_CONSTANT / \
            ((2*M_PI*self.O_atom_mass*BOLTZMANN_CONSTANT*self.temperature)**0.5)
        return ret

    # 初期配置入力
    def pos_init(self):
        with open("Oxy_free.cfg") as f:
            Ti_flag = False
            O_flag = False
            Ti_atom_data_line = 0
            O_atom_data_line = 0
            for s_line in f:
                if "Number of particles" in s_line:
                    self.Ti_atom_num = self.Ti_atom_num
                    self.Ti_x_pos = np.zeros(self.Ti_atom_num, dtype=float)
                    self.Ti_y_pos = np.zeros(self.Ti_atom_num, dtype=float)
                    self.Ti_z_pos = np.zeros(self.Ti_atom_num, dtype=float)
                    self.Ti_gauss_width = np.zeros(
                        self.Ti_atom_num, dtype=float)
                    self.Ti_occupancy = np.zeros(self.Ti_atom_num, dtype=float)
                    self.Ti_atom_mass = np.zeros(self.Ti_atom_num, dtype=float)
                    self.O_atom_num = self.O_atom_num
                    self.O_x_pos = np.zeros(self.O_atom_num, dtype=float)
                    self.O_y_pos = np.zeros(self.O_atom_num, dtype=float)
                    self.O_z_pos = np.zeros(self.O_atom_num, dtype=float)
                    self.O_gauss_width = np.zeros(self.O_atom_num, dtype=float)
                    self.O_occupancy = np.zeros(self.O_atom_num, dtype=float)
                    self.O_atom_mass = np.zeros(self.O_atom_num, dtype=float)
                elif s_line[0:2] == "Cu":
                    Ti_flag = True
                elif Ti_flag and s_line != "":
                    split_s_line = s_line.split()
                    self.Ti_x_pos[Ti_atom_data_line] = float(
                        split_s_line[0])*self.lat_parameter_x
                    self.Ti_y_pos[Ti_atom_data_line] = float(
                        split_s_line[1])*self.lat_parameter_y
                    self.Ti_z_pos[Ti_atom_data_line] = float(
                        split_s_line[2])*self.lat_parameter_z
                    self.Ti_occupancy[Ti_atom_data_line] = 0.99999999
                    self.Ti_gauss_width[Ti_atom_data_line] = 300
                    self.Ti_atom_mass[Ti_atom_data_line] = self.Ti_mass
                    Ti_atom_data_line += 1
                    Ti_flag = False
                elif s_line[0] == "O":
                    O_flag = True
                elif O_flag and s_line != "":
                    split_s_line = s_line.split()
                    self.O_x_pos[O_atom_data_line] = float(
                        split_s_line[0])*self.lat_parameter_x
                    self.O_y_pos[O_atom_data_line] = float(
                        split_s_line[1])*self.lat_parameter_y
                    self.O_z_pos[O_atom_data_line] = float(
                        split_s_line[2])*self.lat_parameter_z
                    if O_atom_data_line == 0:
                        self.O_occupancy[O_atom_data_line] = 0.99999999
                    else:
                        self.O_occupancy[O_atom_data_line] = 0.00000001
                    self.O_gauss_width[O_atom_data_line] = 800
                    self.O_atom_mass[O_atom_data_line] = self.O_mass
                    O_atom_data_line += 1
                    O_flag = False
        f.close()

        self.atom_x_pos = np.concatenate([self.O_x_pos, self.Ti_x_pos])
        self.atom_y_pos = np.concatenate([self.O_y_pos, self.Ti_y_pos])
        self.atom_z_pos = np.concatenate([self.O_z_pos, self.Ti_z_pos])
        self.atom_occupancy = np.concatenate(
            [self.O_occupancy, self.Ti_occupancy])
        self.atom_gauss_width = np.concatenate(
            [self.O_gauss_width, self.Ti_gauss_width])
        self.atom_mass = np.concatenate([self.O_atom_mass, self.Ti_atom_mass])

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

    # 近接リスト作成
    def make_Verlet_neighbor_list(self):
        self.reference_neighbor_list = []
        self.reference_neighbor_list_x_pos = []
        self.reference_neighbor_list_y_pos = []
        self.reference_neighbor_list_z_pos = []
        self.reference_neighbor_list_occ = []
        self.reference_neighbor_list_alpha = []
        nlist = -1      # 0-indexed
        for i in range(self.O_atom_num):
            self.Verlet_neighbor_list[i] = nlist+1
            for j in range(self.atom_num):
                if not i == j:
                    dr2 = self.abs_two_atom_distance(i, j)
                    if dr2 <= self.outer_cutoff**2:
                        nlist += 1
                        self.reference_neighbor_list.append(j)  # 原子number
                        self.reference_neighbor_list_x_pos.append(
                            self.atom_x_pos[j])   # x_pos
                        self.reference_neighbor_list_y_pos.append(
                            self.atom_y_pos[j])   # y_pos
                        self.reference_neighbor_list_z_pos.append(
                            self.atom_z_pos[j])   # z_pos
                        self.reference_neighbor_list_occ.append(
                            self.atom_occupancy[j])   # occupancy
                        self.reference_neighbor_list_alpha.append(
                            self.atom_gauss_width[j])   # alpha
        self.Verlet_neighbor_list[-1] = nlist+1
        self.reference_neighbor_list = np.array(
            self.reference_neighbor_list, dtype=int)
        self.reference_neighbor_list_x_pos = np.array(
            self.reference_neighbor_list_x_pos, dtype=float)
        self.reference_neighbor_list_y_pos = np.array(
            self.reference_neighbor_list_y_pos, dtype=float)
        self.reference_neighbor_list_z_pos = np.array(
            self.reference_neighbor_list_z_pos, dtype=float)
        self.reference_neighbor_list_occ = np.array(
            self.reference_neighbor_list_occ, dtype=float)
        self.reference_neighbor_list_alpha = np.array(
            self.reference_neighbor_list_alpha, dtype=float)

    def culc_VG_energy(self, i):
        start = self.Verlet_neighbor_list[i]
        end = self.Verlet_neighbor_list[i+1]
        if start > end:
            return 0
        i_pos_x = self.atom_x_pos[i]
        i_pos_y = self.atom_y_pos[i]
        i_pos_z = self.atom_z_pos[i]
        i_alpha = self.atom_gauss_width[i]
        j_list = self.reference_neighbor_list[start:end]
        j_list_x = self.reference_neighbor_list_x_pos[start:end]
        j_list_y = self.reference_neighbor_list_y_pos[start:end]
        j_list_z = self.reference_neighbor_list_z_pos[start:end]
        j_list_occ = self.reference_neighbor_list_occ[start:end]
        j_list_alpha = self.reference_neighbor_list_alpha[start:end]
        ret_data = Cc(i, i_pos_x, i_pos_y, i_pos_z, i_alpha, j_list,
                      j_list_x, j_list_y, j_list_z, j_list_occ, j_list_alpha)
        pair_i, embed_i = ret_data.culc_data()
        ret = pair_i + embed_i
        return ret

    def mixed_entropy(self):
        occ = self.O_occupancy
        return occ*np.log(occ) + (1-occ)*np.log(1-occ)

    def culc_all_total_energy(self):
        # pair, embedのリストと総和
        ret_list = Parallel(
            n_jobs=-1)([delayed(self.culc_VG_energy)(i) for i in range(self.O_atom_num)])
        self.pair_and_embed = ret_list
        self.VG_energy_list = self.O_occupancy*np.array(ret_list)
        VG_energy = np.sum(self.VG_energy_list)
        # vib_ent_listの作成と総和
        self.vib_ent_list = np.log(
            (self.O_gauss_width*self.thermal_wavelength()**2)/M_PI)-1
        vib_energy = 1.5*BOLTZMANN_CONSTANT_EV*self.temperature * \
            np.sum(self.O_occupancy*self.vib_ent_list)
        # mixed_ent_listの作成と総和
        self.mixed_list = self.mixed_entropy()
        mixed_energy = BOLTZMANN_CONSTANT_EV * \
            self.temperature*np.sum(self.mixed_list)
        return VG_energy + vib_energy + mixed_energy

    def optimisation_parameter_i(self, i):
        # Tiは不動のものとして扱うので酸素のpositionとgauss_widthを更新できるようにする
        if i < self.O_atom_num:        # i is O
            # ガウス幅
            # base+sigma
            self.atom_gauss_width[i] += self.gauss_sigma
            forward_gauss = self.culc_VG_energy(i)
            # base-sigma
            self.atom_gauss_width[i] -= 2*self.gauss_sigma
            back_gauss = self.culc_VG_energy(i)
            # baseに戻す
            self.atom_gauss_width[i] += self.gauss_sigma
            delta_gauss = (forward_gauss-back_gauss)/(2*self.gauss_sigma)
            # 位置(x)
            self.atom_x_pos[i] += self.pos_sigma
            forward_x = self.culc_VG_energy(i)
            self.atom_x_pos[i] -= 2*self.pos_sigma
            back_x = self.culc_VG_energy(i)
            self.atom_x_pos[i] += self.pos_sigma
            delta_x = (forward_x-back_x)/(2*self.pos_sigma)
            # 位置(y)
            self.atom_y_pos[i] += self.pos_sigma
            forward_y = self.culc_VG_energy(i)
            self.atom_y_pos[i] -= 2*self.pos_sigma
            back_y = self.culc_VG_energy(i)
            self.atom_y_pos[i] += self.pos_sigma
            delta_y = (forward_y-back_y)/(2*self.pos_sigma)
            # 位置(z)
            self.atom_z_pos[i] += self.pos_sigma
            forward_z = self.culc_VG_energy(i)
            self.atom_z_pos[i] -= 2*self.pos_sigma
            back_z = self.culc_VG_energy(i)
            self.atom_z_pos[i] += self.pos_sigma
            delta_z = (forward_z-back_z)/(2*self.pos_sigma)
        return delta_gauss, delta_x, delta_y, delta_z

    def update_parameter(self):
        self.rate = 0.1
        while True:
            ret_list = Parallel(
                n_jobs=-1)([delayed(self.optimisation_parameter_i)(i) for i in range(self.O_atom_num)])
            self.gauss_differential_list = np.array(
                [ret_list[i][0] for i in range(self.O_atom_num)])
            self.x_differential_list = np.array(
                [ret_list[i][1] for i in range(self.O_atom_num)])
            self.y_differential_list = np.array(
                [ret_list[i][2] for i in range(self.O_atom_num)])
            self.z_differential_list = np.array(
                [ret_list[i][3] for i in range(self.O_atom_num)])
            self.atom_gauss_width[:self.O_atom_num] -= self.rate * \
                self.gauss_differential_list
            self.atom_x_pos[:self.O_atom_num] -= self.rate * \
                self.x_differential_list
            self.atom_y_pos[:self.O_atom_num] -= self.rate * \
                self.y_differential_list
            self.atom_z_pos[:self.O_atom_num] -= self.rate * \
                self.z_differential_list
            after_total_energy = self.culc_all_total_energy()
            if after_total_energy < self.current_total_energy:
                self.atom_gauss_width[:self.O_atom_num] += self.rate * \
                    self.gauss_differential_list
                self.atom_x_pos[:self.O_atom_num] += self.rate * \
                    self.x_differential_list
                self.atom_y_pos[:self.O_atom_num] += self.rate * \
                    self.y_differential_list
                self.atom_z_pos[:self.O_atom_num] += self.rate * \
                    self.z_differential_list
                self.rate *= 0.75
            if abs(after_total_energy - self.current_total_energy) < 1e-4:
                self.current_total_energy = after_total_energy
                break
            self.current_total_energy = after_total_energy
        return after_total_energy

    def culc_atomic_formation_energy(self):
        self.form_energy = self.pair_and_embed + 1.5*BOLTZMANN_CONSTANT_EV*self.temperature*(self.vib_ent_list)
        print(self.form_energy)

    def energy_jump(self, i):
        start = self.Verlet_neighbor_list[i]
        end = self.Verlet_neighbor_list[i+1]
        j_list = self.reference_neighbor_list[start:end]
        occ_i = self.atom_occupancy[i]
        ret = 0
        for j in j_list:
            if j < self.O_atom_num:
                occ_j = self.atom_occupancy[j]
                f_ij = self.form_energy[i] - self.form_energy[j]
                gamma_ij = self.diff_coef*math.exp(-(self.delta_Q+f_ij/2)/(BOLTZMANN_CONSTANT_EV*self.temperature))
                gamma_ji = self.diff_coef*math.exp(-(self.delta_Q-f_ij/2)/(BOLTZMANN_CONSTANT_EV*self.temperature))
                ret += occ_j*(1-occ_i)*gamma_ij - occ_i*(1-occ_j)*gamma_ji
        return ret

    # 濃度時間発展
    def update_concentration(self):
        self.culc_atomic_formation_energy()
        ret_list = Parallel(n_jobs=-1)([delayed(self.energy_jump)(i) for i in range(self.O_atom_num)])
        print(ret_list)
        self.atom_occupancy[:self.O_atom_num] += self.time_step*np.array(ret_list)
        print(self.atom_occupancy[:self.O_atom_num])
        if np.all(self.atom_occupancy[:self.O_atom_num] > 0):
            return True
        print("error")
        return False

    def main_loop(self):
        # 初期入力
        self.pos_init()
        # while start
        while True:
            self.make_Verlet_neighbor_list()
            # 自由エネルギー算出
            self.current_total_energy = self.culc_all_total_energy()
            print("cur_energy", self.current_total_energy)
            # エネルギーの最小化
            print(self.update_parameter())
            break
            # 濃度時間変化
            # if not self.update_concentration():
                # break
            # while end
