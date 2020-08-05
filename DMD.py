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
    # lat_x --> x軸の格子点の数
    def __init__(self, lat_parameter, lat_x, lat_y, lat_z, temperature, mass):
        self.lat_parameter = lat_parameter
        self.lat_x = lat_x 
        self.lat_y = lat_y
        self.lat_z = lat_z
        self.atom_number = self.lat_x * self.lat_y * self.lat_z
        self.temperature = temperature
        # 質量のlist
        self.mass = np.zeros(self.atom_number, dtype = float)
        # X_i(3方向)
        self.x_position = np.zeros(self.atom_number, dtype = float)
        self.y_position = np.zeros(self.atom_number, dtype = float)
        self.z_position = np.zeros(self.atom_number, dtype = float)
        # alpha_i
        self.alpha = np.zeros(self.atom_number, dtype = float)
        # c_i
        self.occupancy = np.zeros(self.atom_number, dtype = float)
    
    # 初期状態の入力
    def initial_pos(self):
        i = -1
        for ix in range(self.lat_x):
            for iy in range(self.lat_y):
                for iz in range(self.lat_z):
                    i += 1
                    self.mass[i] = random.random()
                    self.x_position[i] = self.lat_parameter*ix
                    self.y_position[i] = self.lat_parameter*iy
                    self.z_position[i] = self.lat_parameter*iz
                    self.alpha[i] = random.random()
                    self.occupancy[i] = random.random()
    
    #  free_energy (reference:(8))
    def DMD_free_energy(self):
        free_energy = self.Coulomb_interaction() + self.electron_density_dependency() + self.vivrations_entropy() + self.mixed_entropy()
        print(free_energy)
        return free_energy

    # クーロン相互作用
    def Coulomb_interaction(self):
        return 0
    
    # 電子密度依存項
    def electron_density_dependency(self):
        return 0
    
    # 振動のエントロピー項
    # TODO:変数名がわかりにくいので修正
    def vivrations_entropy(self):
        constant = 3 * const.BOLTZMANN_CONSTANT * self.temperature
        viv_ent_list = self.occupancy * (np.log(self.alpha*(self.thermal_wavelength()**2)/const.PI)-1)
        return constant * np.sum(viv_ent_list)
    
    # 混合のエントロピー項
    # TODO:変数名がわかりにくいので修正
    def mixed_entropy(self):
        constant = const.BOLTZMANN_CONSTANT * self.temperature
        mix_ent_list = self.occupancy * np.log(self.occupancy) + (1-self.occupancy) * np.log(1-self.occupancy)
        return constant * np.sum(mix_ent_list)
    
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
