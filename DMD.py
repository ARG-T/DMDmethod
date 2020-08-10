# import
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import const


# DMD Class
class DMDmethod:

    # const
    const.BOLTZMANN_CONSTANT = 1.380649 * (10**(-23))
    const.PI = math.pi
    const.DIRACS_CONSTANT = 1.054571817 * (10**(-34))

    # init
    def __init__(self, lat_parameter, lat_x, lat_y, lat_z, atom_position, temperature, mass, x_period, y_period, z_period):
        # 格子定数
        # TODO : 立方晶以外のにも対応できるようにする
        self.lat_parameter = lat_parameter
        # lat_x --> x軸方向に格子がいくつあるか
        self.lat_x = lat_x
        self.lat_y = lat_y
        self.lat_z = lat_z
        # 単位格子の数
        self.lattice_number = self.lat_x * self.lat_y * self.lat_z
        # 単位格子内の原子の位置
        self.atom_position = atom_position
        self.temperature = temperature
        # 質量のlist
        self.mass = []
        # X_i(3方向)
        self.x_position = []
        self.y_position = []
        self.z_position = []
        # alpha_i
        self.alpha = []
        # c_i
        self.occupancy = []
        # 原子の個数(initial_pos時に決定)
        self.atom_number = 0
        # ベルレリスト
        self.Verlet_neighbor_list = []
        # 参照用リスト
        self.reference_neighbor_list = []
        # カットオフ半径(r_cut)
        self.cutoff_radius = 0
        # リスト半径(r_list)
        self.list_radius = 20.01
        # 周期境界条件
        self.x_period = x_period
        self.y_period = y_period
        self.z_period = z_period

    # 初期状態の入力
    def initial_pos(self):
        for ix in range(self.lat_x+1):
            for iy in range(self.lat_y+1):
                for iz in range(self.lat_z+1):
                    for x_pos, y_pos, z_pos in self.atom_position:
                        x = self.lat_parameter*(ix+x_pos)
                        y = self.lat_parameter*(iy+y_pos)
                        z = self.lat_parameter*(iz+z_pos)
                        # 周期境界条件
                        # 周期境界条件が満たされる場合(True)は格子を一つ加算したぎりぎりまで、
                        # 満たされていない場合(False)は微小量を足す(不等号を用いるので)
                        x_add = 1.0 if self.x_period else 1e-9
                        y_add = 1.0 if self.y_period else 1e-9
                        z_add = 1.0 if self.z_period else 1e-9
                        if (0 <= x < self.lat_parameter*(self.lat_x+x_add) and
                            0 <= y < self.lat_parameter*(self.lat_y+y_add) and
                                0 <= z < self.lat_parameter*(self.lat_z+z_add)):
                            self.x_position.append(x)
                            self.y_position.append(y)
                            self.z_position.append(z)
                            self.mass.append(random.random())
                            self.alpha.append(random.random())
                            self.occupancy.append(random.random())
        self.x_position = np.array(self.x_position, dtype=float)
        self.y_position = np.array(self.y_position, dtype=float)
        self.z_position = np.array(self.z_position, dtype=float)
        self.mass = np.array(self.mass, dtype=float)
        self.alpha = np.array(self.alpha, dtype=float)
        self.occupancy = np.array(self.occupancy, dtype=float)
        self.atom_number = len(self.mass)
        self.Verlet_neighbor_list = [0] * (self.atom_number)
        self.reference_neighbor_list = [0] * (self.atom_number*(self.atom_number-1)//2)

    # 二原子間の距離
    def two_atoms_distance(self, i, j):
        dx = abs(self.x_position[i]-self.x_position[j])
        dy = abs(self.y_position[i]-self.y_position[j])
        dz = abs(self.z_position[i]-self.z_position[j])
        return dx, dy, dz

    # 近接リスト作成
    # 周期境界条件(~period)
    def make_Verlet_neighbor_list(self):
        nlist = -1
        for i in range(self.atom_number):
            self.Verlet_neighbor_list[i] = nlist+1
            for j in range(i, self.atom_number):
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
        # print(self.Verlet_neighbor_list)
        # print(self.reference_neighbor_list)

    # w_ijを求める
    # TODO:積分をどうするか？
    def inter_potential_function(self, pos_i, pos_j, alpha_ij):
        return 0

    # ψ_ijを求める
    # TODO:積分をどうするか？
    def electron_density_function(self, pos_i, pos_j, alpha_ij):
        return 0

    # 近接リストを用いて愚直なO(N^2)より軽くする
    def atomic_interaction(self):
        # 原子iに関する二体間ポテンシャルと有効電子密度関数をいれるリスト
        # numpyはアクセスが微妙なのでlistでアクセスして最後にnumpy.arrayにする
        interbody_potential_list = [0]*(self.atom_number)
        ele_density_list = [0]*(self.atom_number)
        # カットオフ半径内の原子に対してw_ij, psi_ijを求めて足す
        for i in range(self.atom_number-1):
            start = self.Verlet_neighbor_list[i]
            end = self.Verlet_neighbor_list[i+1]
            inter_potential = 0
            if start <= end:
                j_list = self.reference_neighbor_list[start:end]
                for j in j_list:
                    dx, dy, dz = self.two_atoms_distance(i, j)
                    dr = dx**2 + dy**2 + dz**2
                    # 参考論文では距離を求めなおしているが、
                    # 近接リスト作成時にcutoffかどうかも確認できるのでは？
                    # TODO:一旦論文通りに書いてその後最適化させる
                    if dr < self.cutoff_radius**2:
                        alpha_ij = (self.alpha[i]*self.alpha[j]) / (self.alpha[i]+self.alpha[j])
                        inter_potential += self.occupancy[j] * self.inter_potential_function(i, j, alpha_ij)
                        ele_density = self.electron_density_function(i, j, alpha_ij)
                        ele_density_list[i] += ele_density
                        ele_density_list[j] += ele_density
            interbody_potential_list[i] += inter_potential
        return np.sum(self.occupancy * (np.array(interbody_potential_list) + np.array(ele_density_list)))

    # 振動のエントロピー項
    def vivrations_entropy(self):
        viv_ene = 3 * const.BOLTZMANN_CONSTANT * self.temperature
        viv_ent_list = self.occupancy * (np.log(self.alpha*(self.thermal_wavelength()**2)/const.PI)-1)
        return viv_ene * np.sum(viv_ent_list)

    # 混合のエントロピー項
    def mixed_entropy(self):
        mix_ene = const.BOLTZMANN_CONSTANT * self.temperature
        mix_ent_list = self.occupancy * np.log(self.occupancy) + (1-self.occupancy) * np.log(1-self.occupancy)
        return mix_ene * np.sum(mix_ent_list)

    # ドブロイ波長
    def thermal_wavelength(self):
        return const.DIRACS_CONSTANT * ((2*const.PI)/(self.mass*const.BOLTZMANN_CONSTANT*self.temperature))**(1/2)

    #  free_energy (reference:(8))
    def DMD_free_energy(self):
        free_energy = self.atomic_interaction() + self.vivrations_entropy() + self.mixed_entropy()
        # デバッグ用
        # print(free_energy)
        return free_energy

    # メインループ
    def main_roop(self):
        self.initial_pos()
        self.make_Verlet_neighbor_list()
        self.DMD_free_energy()
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
