# import
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# DMD Class
class DMDmethod:

    # init
    # lat_x --> x軸の格子点の数
    def __init__(self, lat_parameter, lat_x, lat_y, lat_z, temperature, mass):
        self.lat_parameter = lat_parameter
        self.lat_x = lat_x 
        self.lat_y = lat_y
        self.lat_z = lat_z
        self.atom_number = self.lat_x * self.lat_y * self.lat_z
        self.temperature = temperature
        self.mass = mass
        self.x_position = np.zeros(self.atom_number, dtype = float)
        self.y_position = np.zeros(self.atom_number, dtype = float)
        self.z_position = np.zeros(self.atom_number, dtype = float)
        self.occupancy = np.zeros(self.atom_number, dtype = float)
    
    # 初期状態の入力
    def initial_pos(self):
        i = -1
        for ix in range(self.lat_x):
            for iy in range(self.lat_y):
                for iz in range(self.lat_z):
                    i += 1
                    self.x_position[i] = self.lat_parameter*ix
                    self.y_position[i] = self.lat_parameter*iy
                    self.z_position[i] = self.lat_parameter*iz
                    self.occupancy[i] = random.random()
    
    #  free_energy (reference:(8))
    def DMD_free_energy(self, alpha, mean_pos, concentration):
        # 第一項 (クーロンポテンシャル項)
        F_DMD_first = 0
        # 第二項 (埋め込みエネルギー項)
        F_DMD_second = 0
        # 第三項 (振動のエントロピー項)
        F_DMD_third = 0
        # 第四項 (混合のエントロピー項)
        F_DMD_fourth = 0
        return F_DMD_first + F_DMD_second + F_DMD_third + F_DMD_fourth
    
    # メインループ
    def main_roop(self):
        self.initial_pos()
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
