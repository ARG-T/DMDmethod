# import
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import const
import csv


# DMD Class
class DMDmethod:

    # const
    const.BOLTZMANN_CONSTANT = 1.380649 * (10**(-23))
    const.PI = math.pi
    const.DIRACS_CONSTANT = 1.054571817 * (10**(-34))

    # init
    # 初期状態も同時にinp
    def __init__(self):
        with open("atom_info.csv") as f:
            reader = csv.reader(f)
            info = [row for row in reader]

        self.lat_parameter = float(info[1][0])
        self.lat_x = int(info[1][1])
        self.lat_y = int(info[1][2])
        self.lat_z = int(info[1][3])
        self.temperature = float(info[1][4])
        self.x_period = info[1][5]
        self.y_period = info[1][6]
        self.z_period = info[1][7]
        self.atom_number = int(info[1][8])
        # カットオフ半径(r_cut)
        self.cutoff_radius = float(info[1][9])
        # リスト半径(r_list)
        self.list_radius = float(info[1][10])
        # 易動度
        self.mobility = float(info[1][11])
        # 活性化エネルギー
        self.activation_energy = float(info[1][12])
        # 時間ステップ
        self.time_step = float(info[1][13])
        # ステップ回数
        self.MAX_STEP = int(info[1][14])
        self.E_1 = float(info[4][0])
        self.E_2 = float(info[4][1])
        self.r_1 = float(info[4][2])
        self.r_2 = float(info[4][3])
        self.a_1 = float(info[4][4])
        self.a_2 = float(info[4][5])
        self.delta = float(info[4][6])
        self.h = float(info[4][8])
        self.r_list = [float(info[4][9]), float(info[4][10]), float(info[4][11])]
        self.S_list = [float(info[4][12]), float(info[4][13]), float(info[4][14])]
        self.a = float(info[4][15])
        self.r_3 = float(info[4][16])
        self.r_4 = float(info[4][17])
        self.b_1 = float(info[4][18])
        self.b_2 = float(info[4][19])
        # ベルレリスト
        self.Verlet_neighbor_list = [0] * self.atom_number
        # 参照用リスト
        self.reference_neighbor_list = [0] * (self.atom_number**2)
        self.x_position = np.zeros(self.atom_number, dtype=float)
        self.y_position = np.zeros(self.atom_number, dtype=float)
        self.z_position = np.zeros(self.atom_number, dtype=float)
        self.mass = np.zeros(self.atom_number, dtype=float)
        self.alpha = np.zeros(self.atom_number, dtype=float)
        self.occupancy = np.zeros(self.atom_number, dtype=float)
        self.Verlet_neighbor_list = [0] * (self.atom_number)
        self.reference_neighbor_list = [
            0] * (self.atom_number*(self.atom_number-1)//2)
        for row_num in range(6, 6+self.atom_number):
            self.x_position[row_num-6] = info[row_num][1]
            self.y_position[row_num-6] = info[row_num][2]
            self.z_position[row_num-6] = info[row_num][3]
            self.mass[row_num-6] = info[row_num][4]
            self.alpha[row_num-6] = info[row_num][5]
            self.occupancy[row_num-6] = info[row_num][6]
        # 化学ポテンシャル
        self.chem_potential = []

    # 二原子間の距離
    def two_atoms_distance(self, i, j):
        dx = abs(self.x_position[i]-self.x_position[j])
        dy = abs(self.y_position[i]-self.y_position[j])
        dz = abs(self.z_position[i]-self.z_position[j])
        return dx, dy, dz

    def two_atoms_dist_value(self, i, j):
        dx, dy, dz = self.two_atoms_distance(i, j)
        return (dx**2 + dy**2 + dz**2) ** (1/2)

    # 近接リスト作成
    # 周期境界条件(~period)
    def make_Verlet_neighbor_list(self):
        nlist = -1
        for i in range(self.atom_number):
            self.Verlet_neighbor_list[i] = nlist+1
            for j in range(i+1, self.atom_number):
                dx, dy, dz = self.two_atoms_distance(i, j)
                # x方向の周期境界条件が適用されている
                if self.x_period:
                    dx = min(dx, self.lat_parameter*(self.lat_x+1)-dx)
                if self.y_period:
                    dy = min(dy, self.lat_parameter*(self.lat_y+1)-dy)
                if self.z_period:
                    dz = min(dz, self.lat_parameter*(self.lat_z+1)-dz)
                dr = dx**2+dy**2+dz**2
                if dr < (self.list_radius)**2:
                    nlist += 1
                    self.reference_neighbor_list[nlist] = j
        self.Verlet_neighbor_list[self.atom_number-1] = nlist+1
        # デバッグ用
        print(self.Verlet_neighbor_list)
        print(self.reference_neighbor_list)

    # w_ijを求める
    # TODO:積分をどうするか？
    def inter_potential_function(self, i, j):
        return random.random()

    # ψ_ijを求める
    # TODO:積分をどうするか？
    def electron_density_function(self, r):
        func = self.a * math.exp(-self.b_1*(r-self.r_3)**2) + math.exp(-self.b_2*(r-self.r_4))
        return func*self.cutoff_function(r)

    def diff_electron_density_function(self, r):
        func1 = (self.a * (-2*self.b_1*(r-self.r_3)) * math.exp(-self.b_1*(r-self.r_3)**2) - self.b_2 * math.exp(-self.b_2 * (r-self.r_4))) * self.cutoff_function(r)
        func2 = (self.a * math.exp(-self.b_1*(r-self.r_3)**2) + math.exp(-self.b_2*(r-self.r_4))) * self.diff_cut_off_function(r)
        return func1 + func2

    # 近接リストを用いて愚直なO(N^2)より軽くする
    # F_DMDの第一項と第二項を求めるのがメインだが、同時に化学ポテンシャルも求める
    def atomic_interaction(self):
        # 原子iに関する二体間ポテンシャルと有効電子密度関数をいれるリスト
        # numpyはアクセスが微妙なのでlistでアクセスして最後にnumpy.arrayにする
        interbody_potential_list = [0]*(self.atom_number)
        ele_density_list = [0]*(self.atom_number)
        self.chem_potential = np.zeros(self.atom_number, dtype=float)
        # カットオフ半径内の原子に対してw_ij, psi_ijを求めて足す
        for i in range(self.atom_number-1):
            start = self.Verlet_neighbor_list[i]
            end = self.Verlet_neighbor_list[i+1]
            inter_potential_sum = 0
            if start <= end:
                j_list = self.reference_neighbor_list[start:end]
                for j in j_list:
                    dx, dy, dz = self.two_atoms_distance(i, j)
                    dr = dx**2 + dy**2 + dz**2
                    # 参考論文では距離を求めなおしているが、
                    # 近接リスト作成時にcutoffかどうかも確認できるのでは？
                    # TODO:一旦論文通りに書いてその後最適化させる
                    if dr < self.cutoff_radius**2:
                        alpha_ij = (
                            self.alpha[i]*self.alpha[j]) / (self.alpha[i]+self.alpha[j])
                        inter_potential = self.inter_potential_function(
                            i, j, alpha_ij)
                        self.atomic_interaction_inf(i, j, inter_potential)
                        inter_potential_sum += self.occupancy[j] * \
                            inter_potential
                        ele_density = self.electron_density_function(
                            i, j, alpha_ij)
                        ele_density_list[i] += ele_density
                        ele_density_list[j] += ele_density
            interbody_potential_list[i] += inter_potential
        interbody_potential_list = np.array(interbody_potential_list)
        ele_density_list = np.array(ele_density_list)
        self.chem_potential += ele_density_list
        return np.sum(self.occupancy * (interbody_potential_list + ele_density_list))

    # 振動のエントロピー項
    # 同時に化学ポテンシャル
    def vivrations_entropy(self):
        viv_ene = 3 * const.BOLTZMANN_CONSTANT * self.temperature
        viv_ent_list = self.occupancy * \
            (np.log(self.alpha*(self.thermal_wavelength()**2)/const.PI)-1)
        self.chem_potential += viv_ent_list
        return viv_ene * np.sum(viv_ent_list)

    # 混合のエントロピー項
    def mixed_entropy(self):
        mix_ene = const.BOLTZMANN_CONSTANT * self.temperature
        mix_ent_list = self.occupancy * \
            np.log(self.occupancy) + (1-self.occupancy) * \
            np.log(1-self.occupancy)
        return mix_ene * np.sum(mix_ent_list)

    # ドブロイ波長
    def thermal_wavelength(self):
        return const.DIRACS_CONSTANT * ((2*const.PI)/(self.mass*const.BOLTZMANN_CONSTANT*self.temperature))**(1/2)

    # free_energy (reference:(8))
    def DMD_free_energy(self):
        free_energy = self.atomic_interaction() + self.vivrations_entropy() + \
            self.mixed_entropy()
        # デバッグ用
        # print(free_energy)
        return free_energy

    def alpha_inter(self, i, j):
        a_ij = (1/self.alpha[i] + 1/self.alpha[j])**(-1)
        return a_ij

    def unit_step_function(self, x):
        return 1 if x >= 1 else 0

    def cutoff_function(self, r):
        x = (r-self.cutoff_radius)/self.h
        return x**4/(1+x**4) if x < 0 else 0

    def diff_cut_off_function(self, r):
        x = (r-self.cutoff_radius)/self.h
        return 4*x**3/(self.h*(1+(x**4))**2)

    # partial(F_DMD)/partial(X_i)
    def diff_DMD_X(self, i):
        return

    def Morse_function(self, r, r_0, alpha):
        return math.exp(-2*alpha*(r-r_0))-2*math.exp(-alpha*(r-r_0))

    # Morse_functionの微分形
    def diff_Morse_function(self, r, r_0, alpha):
        return -2*alpha*(math.exp(-2*alpha*(r-r_0))-math.exp(-alpha*(r-r_0)))

    # 積分する関数1
    def integral_function_1(self, i, j, r, theta):
        X = self.two_atoms_dist_value(i, j)
        diff_xi = (X + math.cos(theta)/(r**2 + X**2 + 2*r*X*math.cos(theta))**(1/2))
        func1 = self.cutoff_function(r)*(self.diff_Morse_function(r, self.r_1, self.a_1)*self.E_1 + self.diff_Morse_function(r, self.r_2, self.a_2)*self.E_2)
        func2 = (self.Morse_function(r, self.r_1, self.a_1)*self.E_1 + self.Morse_function(r, self.r_2, self.a_2)*self.E_2 + self.delta) * self.diff_unit_step_function(r)
        func3 = 0
        for r_i, S_i in zip(self.r_list, self.S_list):
            func3 += 4*self.unit_step_function(r_i-r)*S_i*(r_i-r)**3
        return (func1 + func2 - func3) * diff_xi * math.exp(-self.alpha_inter(i, j)*r**2) * r**2 * math.sin(theta)

    # 積分する関数2
    def integral_function_2(self, i, j, r, theta):
        X = self.two_atoms_dist_value(i, j)
        diff_xi = (X + math.cos(theta)/(r**2 + X**2 + 2*r*X*math.cos(theta))**(1/2))
        return diff_xi * self.diff_electron_density_function(r)

    # 積分(モンテカルロ法)
    def Monte_Carlo_integral(self, i, j, r, theta, integral_function):
        ret = 0
        trial = 1000
        for i in range(trial):
            r_rand = self.cutoff_radius * random.random()
            theta_rand = 2 * const.PI * random.random()
            ret += integral_function(i, j, r_rand, theta_rand)
        ret *= self.cutoff_radius * const.PI
        ret /= trial
        return ret

    # 化学ポテンシャルの第一項用
    def atomic_interaction_inf(self, i, j, inter_potential):
        self.chem_potential[i] += self.occupancy[j] * inter_potential
        self.chem_potential[j] += self.occupancy[i] * inter_potential
        return

    # 原子形成エネルギー
    # μの4項目がちょうど打ち消し合う
    def formation_energy(self):
        return self.chem_potential

    # ジャンプ頻度
    def jump_frequency(self, f_i, f_j):
        f_ij = (f_i-f_j)/2
        gamma_ij = self.mobility * \
            math.exp(-(self.activation_energy+f_ij) /
                     (const.BOLTZMANN_CONSTANT*self.temperature))
        gamma_ji = self.mobility * \
            math.exp(-(self.activation_energy-f_ij) /
                     (const.BOLTZMANN_CONSTANT*self.temperature))
        return gamma_ij, gamma_ji

    # 濃度変化
    def occupancy_change(self):
        atom_formation_energy = self.formation_energy()
        occupancy_delta = np.zeros(self.atom_number, dtype=float)
        for i in range(self.atom_number-1):
            i_occ = self.occupancy[i]
            start = self.Verlet_neighbor_list[i]
            end = self.Verlet_neighbor_list[i+1]
            if start <= end:
                j_list = self.reference_neighbor_list[start:end]
                for j in j_list:
                    dx, dy, dz = self.two_atoms_distance(i, j)
                    dr = dx**2 + dy**2 + dz**2
                    if dr < self.cutoff_radius**2:
                        j_occ = self.occupancy[j]
                        gamma_ij, gamma_ji = self.jump_frequency(
                            atom_formation_energy[i], atom_formation_energy[j])
                        delta = j_occ*(1-i_occ)*gamma_ji - \
                            i_occ*(1-j_occ)*gamma_ij
                        occupancy_delta[i] += delta
                        occupancy_delta[j] -= delta
        occupancy_delta *= self.time_step
        self.occupancy += occupancy_delta

    # メインループ
    def main_roop(self):
        self.make_Verlet_neighbor_list()
        for step in range(self.MAX_STEP):
            self.DMD_free_energy()
            self.occupancy_change()
        self.sample_print()

    def sample_print(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        x = self.x_position
        y = self.y_position
        z = self.z_position
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.scatter3D(x, y, z)
        plt.show()
        plt.close()
