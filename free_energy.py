import math
import random
import numpy as np


class FreeEnergy:

    def __init__(self):
        self.rc = 5.50679
        self.h = 0.50037
        self.a = 3.80362
        self.b1 = 0.17394
        self.b2 = 5.25661*10**2
        self.E_list = np.array([2.01458*10**2, 6.59288*10**(-3)])
        self.delta = 0.86225*10**(-2)
        self.alpha_list = np.array([2.97758, 1.54927])
        self.r0_list_bef = np.array([0.83591, 4.46867])
        self.r0_list_aft = np.array([-2.19885, -2.61984*10**2])
        self.S_list = np.array([4.00, 40.00, 1.15*10**3])
        self.rs_list = np.array([2.24, 1.80, 1.20])
        self.F0 = -2.28235
        self.F2 = 1.35535
        self.qn_list = np.array([-1.27775, -0.86074, 1.78804, 2.97571])
        self.Q1 = 0.4000
        self.Q2 = 0.3000
        self.lat_parameter = 4.05
        self.atom_num = 3**3
        self.x_pos = np.zeros(self.atom_num, dtype=float)
        self.y_pos = np.zeros(self.atom_num, dtype=float)
        self.z_pos = np.zeros(self.atom_num, dtype=float)
        self.total_energy = 0
        self.occupancy = np.ones(self.atom_num, dtype=float)

    def pos_init(self):
        i = 0
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    self.x_pos[i] = self.lat_parameter*x
                    self.y_pos[i] = self.lat_parameter*y
                    self.z_pos[i] = self.lat_parameter*z
                    i += 1

    def two_atom_dis(self, i, j):
        dx = self.x_pos[i] - self.x_pos[j]
        dy = self.y_pos[i] - self.y_pos[j]
        dz = self.z_pos[i] - self.z_pos[j]
        return dx, dy, dz

    def abs_two_atom_dis(self, i, j):
        dx, dy, dz = self.two_atom_dis(i, j)
        return (dx**2+dy**2+dz**2)**0.5

    def unit_step_function(self, x):
        return 1.0 if x >= 1 else 0.0

    def Morse_function(self, r, r_0, a):
        ret = math.exp(-2*a*(r-r_0))-2*math.exp(-a*(r-r_0))
        return ret

    def cutoff_function(self, x):
        if x >= 0:
            return 0.0
        else:
            return x**4/(1+x**4)

    def alpha_int(self, i, j):
        return (1/self.occupancy[i] + 1/self.occupancy[j])**(-1)

    # 二体間ポテンシャル
    def pair_potential(self, r):
        ret = 0
        for E, r0, alpha in zip(self.E_list, self.r0_list_bef, self.alpha_list):
            ret += E*self.Morse_function(r, r0, alpha)
        ret += self.delta
        ret *= self.cutoff_function((r-self.rc)/self.h)
        for rs, Sn in zip(self.rs_list, self.S_list):
            ret -= self.unit_step_function(rs-r)*Sn*(rs-r)**4
        return ret

    # 電子密度関数
    def electron_density_function(self, r):
        func1 = self.a*math.exp(-self.b1*(r-self.r0_list_aft[0])**2)
        func2 = math.exp(-self.b2*(r-self.r0_list_aft[1]))
        return (func1+func2)*self.cutoff_function((r-self.rc)/self.h)

    def change_coo(self, r, x, theta):
        return r**2+x**2+2*r*x*math.cos(theta)

    def change_func(self, function, r, X, theta, alpha):
        return function(self.change_coo(r, X, theta))*(r**2)*2*math.sin(theta)*math.exp(-alpha*(r**2))

    # 積分
    def integral_sympson(self, function, i, j):
        n, m = 50, 50
        dx = self.rc/(2*n)
        dy = math.pi/(2*m)
        s = 0
        X = self.abs_two_atom_dis(i, j)
        alpha = self.alpha_int(i, j)
        # r
        for i in range(n):
            x = 2*i*dx
            x1 = x+dx
            x2 = x+2*dx
            sy1, sy2, sy3 = 0, 0, 0
            # theta
            for j in range(m):
                y = 2*j*dy
                y1 = y+dy
                y2 = y+2*dy

                s1 = ((self.change_func(function, x, X, y, alpha) + 4 * self.change_func(function, x, X, y1, alpha) + self.change_func(function, x, X, y2, alpha))/3)*dy
                s2 = ((self.change_func(function, x1, X, y, alpha) + 4 * self.change_func(function, x1, X, y1, alpha) + self.change_func(function, x1, X, y2, alpha))/3)*dy
                s3 = ((self.change_func(function, x2, X, y, alpha) + 4 * self.change_func(function, x2, X, y1, alpha) + self.change_func(function, x2, X, y2, alpha))/3)*dy

                sy1 += s1
                sy2 += s2
                sy3 += s3

            sx = ((sy1+4*sy2+sy3)/3)*dx
            s += sx

        return s

    def integral_monte_carlo(self, function, i, j):
        s = 0
        X = self.abs_two_atom_dis(i, j)
        alpha = self.alpha_int(i, j)
        STEP = 10000
        for _ in range(STEP):
            x, y = random.random()*self.rc, random.random()*math.pi
            s += self.change_func(function, x, X, y, alpha)*self.rc*math.pi

        return s/STEP

    def culc_rho(self, i):
        ret = 0.0
        for j in range(self.atom_num):
            if i == j:
                continue
            else:
                ret += self.occupancy[j]*2*math.pi*((self.alpha_int(i, j)/math.pi)**1.5)*self.integral_sympson(self.electron_density_function, i, j)
        return ret

    def embedding_function(self, rho):
        ret = 0.0
        if rho <= 1:
            ret += self.F0 + (self.F2*(rho-1)**2)*0.5
            for i, qn in enumerate(self.qn_list):
                ret += qn*(rho-1)**(i+3)
        else:
            ret += (self.F0 + (self.F2*(rho-1)**2)*0.5 + self.qn_list[0]*(rho-1)**3 + (self.Q1*(rho-1)**4)/(1+self.Q2*(rho-1)**3))
        return ret

    def culc_total_energy(self):
        pair, embed = 0.0, 0.0
        for i in range(self.atom_num):
            for j in range(self.atom_num):
                if i == j:
                    continue
                pair += self.occupancy[i]*self.occupancy[j]*(2*math.pi*((self.alpha_int(i, j)/math.pi)**1.5)*self.integral_sympson(self.pair_potential, i, j))
            rho = self.culc_rho(i)
            embed += self.occupancy[i]*self.embedding_function(rho)
        print(pair)
        print(embed)
        self.total_energy = pair*0.5 + embed

    def main_loop(self):
        self.pos_init()
        self.culc_total_energy()
        print(self.total_energy)
