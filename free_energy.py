import math
# import random
import numpy as np
import const
import csv


class FreeEnergy:

    # const
    const.BOLTZMANN_CONSTANT = 8.6173336 * (10**(-5))   # eV/K
    const.PI = math.pi
    const.DIRACS_CONSTANT = 6.5821198 * (10**(-16))     # eV*s

    def __init__(self):
        self.mass = 63.5*1.66054*(10**(-27))    # kg
        self.rc = 5.50679   # オングストローム(以下A)
        self.h = 0.50037    # A
        self.a = 3.80362    # None
        self.b1 = 0.17394   # /A^2
        self.b2 = 5.35661*10**2     # /A^1
        self.E_list = [2.01458*10**2, 6.59288*10**(-3)]     # eV
        self.delta = 0.86225*10**(-2)   # A
        self.alpha_list = [2.97758, 1.54927]    # /A
        self.r0_list_bef = [0.83591, 4.46867]   # A
        self.r0_list_aft = [-2.19885, -2.61984*10**2]   # A
        self.S_list = [4.00, 40.00, 1.15*10**3]     # eV/A^4
        self.rs_list = [2.24, 1.80, 1.20]   # A
        self.F0 = -2.28235  # eV
        self.F2 = 1.35535   # eV
        self.qn_list = [-1.27775, -0.86074, 1.78804, 2.97571]   # eV
        self.Q1 = 0.4000    # None
        self.Q2 = 0.3000    # None
        self.lat_parameter = 3.61496    # A
        self.atom_num = 4   # None
        self.x_pos = np.zeros(self.atom_num, dtype=float)   # A
        self.y_pos = np.zeros(self.atom_num, dtype=float)   # A
        self.z_pos = np.zeros(self.atom_num, dtype=float)   # A
        self.gauss_width = np.zeros(self.atom_num, dtype=float)     # /A^2
        self.total_energy = 0   # eV
        self.sigma = 0.00001    # None
        self.temperature = 10   # K
        self.occupancy = np.zeros(self.atom_num, dtype=float)   # None
        self.energy_list = np.zeros(self.atom_num, dtype=float)     # eV
        self.rate = 0.5     # None

    # ドブロイ波長 (A)
    def thermal_wavelength(self):
        return 10**10*const.DIRACS_CONSTANT*(2*const.PI/(self.mass*const.BOLTZMANN_CONSTANT*self.temperature))**0.5

    # 初期配置入力
    def pos_init(self):
        with open("FCC.csv") as f:
            reader = csv.reader(f)
            info = [row for row in reader]

        for i in range(1, self.atom_num+1):
            self.x_pos[i-1] = float(info[i][1])
            self.y_pos[i-1] = float(info[i][2])
            self.z_pos[i-1] = float(info[i][3])
            self.occupancy[i-1] = float(info[i][4])
            self.gauss_width[i-1] = float(info[i][5])

    # 同等の位置を考慮しつつ二つの原子の距離(A)を返す
    def abs_two_atom_dis_list(self, i, j):
        dis_list = []
        x_pos_i, y_pos_i, z_pos_i = self.x_pos[i], self.y_pos[i], self.z_pos[i]
        x_pos_j, y_pos_j, z_pos_j = self.x_pos[j], self.y_pos[j], self.z_pos[j]
        # cutoff半径が格子定数の3倍以内なので余裕をもってx,y,z方向に5つずれた格子まで考える
        for x in range(-5, 6):
            for y in range(-5, 6):
                for z in range(-5, 6):
                    dx = x_pos_i - (x_pos_j + x*self.lat_parameter)
                    dy = y_pos_i - (y_pos_j + y*self.lat_parameter)
                    dz = z_pos_i - (z_pos_j + z*self.lat_parameter)
                    dr_2 = dx**2+dy**2+dz**2
                    if 0 < dr_2 <= self.rc**2:
                        dis_list.append(dr_2**(0.5))
        return dis_list

    # Morse function(unit : None)
    def Morse_function(self, r, r_0, a):
        exp_val = math.exp(-a*(r-r_0))
        ret = exp_val**2-2*exp_val
        return ret

    # cutoff function (unit:None)
    def cutoff_function(self, x):
        if x >= 0:
            return 0.0
        else:
            return (x**4)/(1+x**4)

    # DMD原著(12) (/A^2)
    def alpha_int(self, i, j):
        return (1/self.gauss_width[i] + 1/self.gauss_width[j])**(-1)

    # 二体間ポテンシャル
    def pair_potential(self, r):
        ret = 0
        for E, r0, alpha in zip(self.E_list, self.r0_list_bef, self.alpha_list):
            ret += E*self.Morse_function(r, r0, alpha)
        ret += self.delta
        ret *= self.cutoff_function((r-self.rc)/self.h)
        for rs, Sn in zip(self.rs_list, self.S_list):
            if rs >= r:
                ret -= Sn*(rs-r)**4
        return ret

    # 電子密度関数
    def electron_density_function(self, r):
        func1 = self.a*math.exp(-self.b1*(r-self.r0_list_aft[0])**2)
        func2 = math.exp(-self.b2*(r-self.r0_list_aft[1]))
        return (func1+func2)*self.cutoff_function((r-self.rc)/self.h)

    def change_coo(self, r, x, theta):
        return (r**2+x**2+2*r*x*math.cos(theta))**0.5

    def change_func(self, function, r, X, theta, alpha):
        return function(self.change_coo(r, X, theta))*(r**2)*math.sin(theta)*math.exp(-alpha*(r**2))

    # 積分
    def integral_sympson(self, function, i, j):
        n, m = 50, 50
        dx = self.rc/(2*n)
        dy = const.PI/(2*m)
        s = 0
        X_list = self.abs_two_atom_dis_list(i, j)
        alpha = self.alpha_int(i, j)
        for X in X_list:
            # r
            for i_r in range(n):
                x = 2*i_r*dx
                x1 = x+dx
                x2 = x+2*dx
                sy1, sy2, sy3 = 0, 0, 0
                # theta
                for j_theta in range(m):
                    y = 2*j_theta*dy
                    y1 = y+dy
                    y2 = y+2*dy

                    s1 = (self.change_func(function, x, X, y, alpha) + 4 * self.change_func(function, x, X, y1, alpha) + self.change_func(function, x, X, y2, alpha))*dy/3
                    s2 = (self.change_func(function, x1, X, y, alpha) + 4 * self.change_func(function, x1, X, y1, alpha) + self.change_func(function, x1, X, y2, alpha))*dy/3
                    s3 = (self.change_func(function, x2, X, y, alpha) + 4 * self.change_func(function, x2, X, y1, alpha) + self.change_func(function, x2, X, y2, alpha))*dy/3

                    sy1 += s1
                    sy2 += s2
                    sy3 += s3

                sx = (sy1+4*sy2+sy3)*dx/3
                s += sx

        return s

    """
    def integral_monte_carlo(self, function, i, j):
        s = 0
        X = self.abs_two_atom_dis(i, j)
        alpha = self.alpha_int(i, j)
        STEP = 10000
        for _ in range(STEP):
            x, y = random.random()*self.rc, random.random()*math.pi
            s += self.change_func(function, x, X, y, alpha)*self.rc*math.pi

        return s/STEP
    """

    def culc_rho(self, i):
        ret = 0.0
        for j in range(self.atom_num):
            ret += self.occupancy[j]*2*const.PI*((self.alpha_int(i, j)/const.PI)**1.5)*self.integral_sympson(self.electron_density_function, i, j)
        return ret

    def embedding_function(self, rho):
        ret = 0.0
        if rho <= 1:
            ret += self.F0 + (self.F2*(rho-1)**2)*0.5
            for i, qn in enumerate(self.qn_list):
                ret += qn*(rho-1)**(i+3)
        else:
            ret += (self.F0 + (self.F2*(rho-1)**2)*0.5 + self.qn_list[0]*(rho-1)**3 + self.Q1*(rho-1)**4)/(1+self.Q2*(rho-1)**3)
        return ret

    # 勾配計算時に混合のエントロピーを呼び出さないようにする
    def culc_VG_energy(self, i):
        pair = 0.0
        for j in range(self.atom_num):
            pair += self.occupancy[j]*(2*const.PI*((self.alpha_int(i, j)/const.PI)**1.5)*self.integral_sympson(self.pair_potential, i, j))
        rho = self.culc_rho(i)
        embed = self.embedding_function(rho)
        vib_entropy = math.log(self.gauss_width[i]*(self.thermal_wavelength()**2)/const.PI)-1
        return self.occupancy[i]*(pair*0.5 + embed + 1.5*const.BOLTZMANN_CONSTANT*self.temperature*vib_entropy)

    def mixed_entropy(self, i):
        occ_i = self.occupancy[i]
        ret = 0
        if occ_i != 0:
            ret += occ_i*math.log(occ_i)
        if occ_i != 1:
            ret += (1-occ_i)*math.log(1-occ_i)
        return const.BOLTZMANN_CONSTANT*self.temperature*ret

    def culc_total_energy(self, i):
        return self.culc_VG_energy(i) + self.mixed_entropy(i)

    def culc_all_total_energy(self):
        for i in range(self.atom_num):
            self.energy_list[i] = self.culc_total_energy(i)
        return np.sum(self.energy_list)

    # 中心差分により勾配を求める
    def make_differential_list(self):
        gauss_differential_list = np.zeros(self.atom_num, dtype=float)
        x_differential_list = np.zeros(self.atom_num, dtype=float)
        y_differential_list = np.zeros(self.atom_num, dtype=float)
        z_differential_list = np.zeros(self.atom_num, dtype=float)
        for i in range(self.atom_num):
            # ガウス幅
            # base+sigma
            self.gauss_width[i] += self.sigma
            forward_gauss = self.culc_VG_energy(i)
            # base-sigma
            self.gauss_width[i] -= 2*self.sigma
            back_gauss = self.culc_VG_energy(i)
            gauss_differential_list[i] = (forward_gauss-back_gauss)/(2*self.sigma)
            # baseに戻す
            self.gauss_width[i] += self.sigma
            # 位置(x)
            self.x_pos[i] += self.sigma
            forward_x = self.culc_VG_energy(i)
            self.x_pos[i] -= 2*self.sigma
            back_x = self.culc_VG_energy(i)
            x_differential_list[i] = (forward_x-back_x)/(2*self.sigma)
            self.x_pos[i] += self.sigma
            # 位置(y)
            self.y_pos[i] += self.sigma
            forward_y = self.culc_VG_energy(i)
            self.y_pos[i] -= 2*self.sigma
            back_y = self.culc_VG_energy(i)
            y_differential_list[i] = (forward_y-back_y)/(2*self.sigma)
            self.y_pos[i] += self.sigma
            # 位置(z)
            self.z_pos[i] += self.sigma
            forward_z = self.culc_VG_energy(i)
            self.z_pos[i] -= 2*self.sigma
            back_z = self.culc_VG_energy(i)
            z_differential_list[i] = (forward_z-back_z)/(2*self.sigma)
            self.z_pos[i] += self.sigma
        return gauss_differential_list, x_differential_list, y_differential_list, z_differential_list

    def update_info(self):
        while True:
            gauss_differential_list, x_differential_list, y_differential_list, z_differential_list = self.make_differential_list()
            self.gauss_width -= self.rate*gauss_differential_list
            self.x_pos -= self.rate*x_differential_list
            self.y_pos -= self.rate*y_differential_list
            self.z_pos -= self.rate*z_differential_list
            print(self.rate, self.gauss_width, self.x_pos, self.y_pos, self.z_pos)
            after_total_energy = self.culc_all_total_energy()
            print(after_total_energy)
            if after_total_energy > self.current_total_energy:
                self.gauss_width += self.rate*gauss_differential_list
                self.x_pos += self.rate*x_differential_list
                self.y_pos += self.rate*y_differential_list
                self.z_pos += self.rate*z_differential_list
                self.rate *= 0.9
                print("continue")
                continue
            if abs(after_total_energy-self.current_total_energy) < 0.001:
                print("break")
                break
            print("normal")
            self.current_total_energy = after_total_energy
        return after_total_energy

    # 中心差分
    def culc_chem_potential(self):
        chem_differential_list = np.zeros(self.atom_num, dtype=float)
        for i in range(self.atom_num):
            self.occupancy += self.sigma
            forward_occ = self.culc_all_total_energy()
            self.occupancy -= 2*self.sigma
            back_occ = self.culc_all_total_energy()
            chem_differential_list[i] = (forward_occ-back_occ)/(2*self.sigma)
        return chem_differential_list

    def main_loop(self):
        self.pos_init()
        self.current_total_energy = self.culc_all_total_energy()
        print(self.current_total_energy)
        # self.update_info()
