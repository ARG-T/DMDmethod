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
    # lat_x --> x軸方向に格子がいくつあるか
    # lat_x * lat_y * lat_z * (単位格子内の原子数)
    def __init__(self, lat_parameter, lat_x, lat_y, lat_z, atom_position, temperature, mass):
        # 格子定数
        # TODO : 立方晶以外のにも対応できるようにする
        self.lat_parameter = lat_parameter
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

    # 初期状態の入力
    def initial_pos(self):
        for ix in range(self.lat_x+1):
            for iy in range(self.lat_y+1):
                for iz in range(self.lat_z+1):
                    for x_pos, y_pos, z_pos in self.atom_position:
                        x = self.lat_parameter*(ix+x_pos)
                        y = self.lat_parameter*(iy+y_pos)
                        z = self.lat_parameter*(iz+z_pos)
                        # 範囲内に存在する
                        if (0 <= x <= self.lat_parameter*self.lat_x and
                            0 <= y <= self.lat_parameter*self.lat_y and
                                0 <= z <= self.lat_parameter*self.lat_z):
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

    #  free_energy (reference:(8))
    def DMD_free_energy(self):
        free_energy = self.Coulomb_interaction() \
                    + self.electron_density_dependency() \
                    + self.vivrations_entropy() \
                    + self.mixed_entropy()
        print(free_energy)
        return free_energy

    # クーロン相互作用
    def Coulomb_interaction(self):
        return 0

    # 電子密度依存項
    def electron_density_dependency(self):
        return 0

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

    # メインループ
    def main_roop(self):
        self.initial_pos()
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
