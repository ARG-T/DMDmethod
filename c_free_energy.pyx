import math
import random
import numpy as np
cimport numpy as np
import csv

ctypedef np.float64_t DOUBLE_t

cdef class FreeEnergy(object):

    cdef public double mass
    cdef public double rc
    cdef public double h
    cdef public double a
    cdef public double b1
    cdef public double b2
    cdef public double delta
    cdef public double F0
    cdef public double F2
    cdef public double Q1
    cdef public double Q2
    cdef public double lat_parameter
    cdef public double current_
    cdef public double temperature
    cdef public double sigma
    cdef public double BOLTZMANN_CONSTANT
    cdef public double PI
    cdef public double DIRACS_CONSTANT
    cdef public double current_total_energy
    cdef public double activation_energy
    cdef public double frequency_factor
    cdef public double time_step
    cdef public double rate
    cdef public int atom_num
    cdef public list E_list
    cdef public list alpha_list
    cdef public list r0_list_aft
    cdef public list r0_list_bef
    cdef public list S_list
    cdef public list rs_list
    cdef public list qn_list
    cdef public double[:] x_pos
    cdef public double[:] y_pos
    cdef public double[:] z_pos
    cdef public double[:] occupancy
    cdef public double[:] gauss_width
    cdef public list energy_list
    cdef public double[:] chem_differential_list

    def __init__(self):
        self.mass = 1.0544429e-25
        self.rc = 5.50679
        self.h = 0.50037
        self.a = 3.80362
        self.b1 = 0.17394
        self.b2 = 5.35661*10**2
        self.E_list = [2.01458*10**2, 0.00659288]
        self.delta = 0.0086225
        self.alpha_list = [2.97758, 1.54927]
        self.r0_list_bef = [0.83591, 4.46867]
        self.r0_list_aft = [-2.19885, -2.61984*10**2]
        self.S_list = [4.00, 40.00, 1.15*10**3]
        self.rs_list = [2.24, 1.80, 1.20]
        self.F0 = -2.28235
        self.F2 = 1.35535
        self.qn_list = [-1.27775, -0.86074, 1.78804, 2.97571]
        self.Q1 = 0.4000
        self.Q2 = 0.3000
        self.lat_parameter = 3.61496
        self.atom_num = 4
        self.x_pos = np.zeros(self.atom_num,dtype=float)
        self.y_pos = np.zeros(self.atom_num,dtype=float)
        self.z_pos = np.zeros(self.atom_num,dtype=float)
        self.gauss_width = np.zeros(self.atom_num,dtype=float)
        self.current_total_energy = 0
        self.sigma = 0.00001
        self.temperature = 10.0
        self.occupancy = np.zeros(self.atom_num,dtype=float)
        self.energy_list = [0.0] * self.atom_num
        self.BOLTZMANN_CONSTANT = 8.6173336e-5
        self.PI = math.pi
        self.DIRACS_CONSTANT = 6.5821198e-16
        self.chem_differential_list = np.zeros(self.atom_num,dtype=float)
        # TODO:初期値
        self.activation_energy = 1.0
        # TODO:初期値
        self.frequency_factor = 1.0
        self.time_step = 1.0
        self.rate = 0.1

    # ドブロイ波長
    cdef thermal_wavelength(self):
        return <double>10**10*self.DIRACS_CONSTANT * (2*self.PI/(self.mass*self.BOLTZMANN_CONSTANT*self.temperature))**0.5

    cdef pos_init(self):
        cdef int i
        cdef list info, row
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
    cdef abs_two_atom_dis_list(self, int i, int j):
        cdef list dis_list
        cdef double x_pos_i, y_pos_i, z_pos_i, x_pos_j, y_pos_j, z_pos_j
        cdef double dx, dy, dz, dr_2
        cdef int x, y, z
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
                        dis_list.append(dr_2**0.5)
        return dis_list

    cdef Morse_function(self, double r, double r_0, double a):
        cdef double exp_val, ret
        exp_val = math.exp(-a*(r-r_0))
        ret = exp_val**2-2*exp_val
        return ret

    cdef cutoff_function(self, double x):
        if x >= 0:
            return 0.0
        else:
            return <double>x**4/(1+x**4)

    cdef alpha_int(self, int i, int j):
        cdef double o_i, o_j
        o_i = self.gauss_width[i]
        o_j = self.gauss_width[j]
        return o_i*o_j/(o_i+o_j)

    # 二体間ポテンシャル
    cpdef pair_potential(self, double r):
        cdef double ret, E, r0, alpha, rs, Sn
        ret = self.delta
        for E, r0, alpha in zip(self.E_list, self.r0_list_bef, self.alpha_list):
            ret += E*self.Morse_function(r, r0, alpha)
        ret *= self.cutoff_function(<double>(r-self.rc)/self.h)
        for rs, Sn in zip(self.rs_list, self.S_list):
            if rs >= r:
                ret -= Sn*(rs-r)**4
        return ret

    # 電子密度関数
    cpdef electron_density_function(self, double r):
        cdef double func1, func2
        func1 = self.a*math.exp(-self.b1*(r-self.r0_list_aft[0])**2)
        func2 = math.exp(-self.b2*(r-self.r0_list_aft[1]))
        return (func1+func2)*self.cutoff_function(<double>(r-self.rc)/self.h)

    cdef change_coo(self, double r, double x, double theta):
        return <double>(r**2+x**2+2*r*x*math.cos(theta))**0.5

    cdef change_func(self, function, double r, double X, double theta, double alpha):
        return <double>function(self.change_coo(r, X, theta))*(r**2)*math.sin(theta)*math.exp(-alpha*(r**2))

    # 積分
    cdef integral_sympson(self, function, int i, int j):
        cdef int n, m, i_r, j_theta
        cdef double dx, dy, s, X, alpha, x, x1, x2, sy1, sy2, sy3, y, y1, y2, sx
        n, m = 50, 50
        dx = self.rc/(2*n)
        dy = self.PI/(2*m)
        s = 0.0
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

    """"
        cdef integral_monte_carlo(self, function, int i, int j):
        cdef int STEP, _
        cdef double s, X, alpha, x, y
        s = 0.0
        X = self.abs_two_atom_dis(i, j)
        alpha = self.alpha_int(i, j)
        STEP = 10000
        for _ in range(STEP):
            x, y = random.random()*self.rc, random.random()*math.pi
            s += self.change_func(function, x, X, y, alpha)*self.rc*math.pi

        return <double>s/STEP
    """

    cdef culc_rho(self, int i):
        cdef int j
        cdef double ret
        ret = 0.0
        for j in range(self.atom_num):
            ret += <double>self.occupancy[j]*2*(self.alpha_int(i, j)**1.5)*self.integral_sympson(self.electron_density_function, i, j)/self.PI**0.5
        return ret

    cdef embedding_function(self, double rho):
        cdef int i
        cdef double ret, qn
        ret = 0.0
        if rho <= 1:
            ret += self.F0 + (self.F2*(rho-1)**2)*0.5
            for i, qn in enumerate(self.qn_list):
                ret += qn*(rho-1)**(i+3)
        else:
            ret += self.F0 + (self.F2*(rho-1)**2)*0.5 + self.qn_list[0]*(rho-1)**3 + <double>(self.Q1*(rho-1)**4)/(1+self.Q2*(rho-1)**3)
        return ret

    cdef culc_VG_energy(self, int i):
        cdef int j
        cdef double pair, embed, vib_entropy
        pair = 0.0
        for j in range(self.atom_num):
            pair += <double>self.occupancy[j]*2*(self.alpha_int(i, j)**1.5)*self.integral_sympson(self.pair_potential, i, j)/self.PI**0.5
        rho = self.culc_rho(i)
        embed = self.embedding_function(rho)
        vib_entropy = math.log(self.gauss_width[i]*(self.thermal_wavelength()**2)/self.PI)-1
        return self.occupancy[i]*(pair*0.5 + embed + 1.5*self.BOLTZMANN_CONSTANT*self.temperature*vib_entropy)

    cdef mixed_entropy(self, int i):
        cdef double occ_i, ret
        occ_i = self.occupancy[i]
        ret = 0
        if occ_i != 0:
            ret += occ_i*math.log(occ_i)
        if occ_i != 1:
            ret += (1-occ_i)*math.log(1-occ_i)
        return self.BOLTZMANN_CONSTANT*self.temperature*ret

    cdef culc_total_energy(self, int i):
        return <double>self.culc_VG_energy(i) + <double>self.mixed_entropy(i)
    
    cdef culc_all_total_energy(self):
        cdef int i
        for i in range(self.atom_num):
            self.energy_list[i] = self.culc_total_energy(i)
        return <double>np.sum(self.energy_list)

    # 中心差分により勾配を求める
    cdef make_differential_list(self):
        cdef int i
        cdef double forward_gauss, back_gauss, forward_x, back_x, forward_y, back_y, forward_z, back_z
        cdef np.ndarray[DOUBLE_t, ndim=1] gauss_differential_list, x_differential_list, y_differential_list, z_differential_list
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

    cdef update_info(self):
        cdef double after_total_energy
        cdef np.ndarray[DOUBLE_t, ndim=1] gauss_differential_list, x_differential_list, y_differential_list, z_differential_list
        while True:
            gauss_differential_list, x_differential_list, y_differential_list, z_differential_list = self.make_differential_list()
            self.gauss_width -= self.rate*gauss_differential_list
            self.x_pos -= self.rate*x_differential_list
            self.y_pos -= self.rate*y_differential_list
            self.z_pos -= self.rate*z_differential_list
            after_total_energy = self.culc_all_total_energy()
            print(after_total_energy)
            if after_total_energy > self.current_total_energy:
                self.gauss_width += self.rate*gauss_differential_list
                self.x_pos += self.rate*x_differential_list
                self.y_pos += self.rate*y_differential_list
                self.z_pos += self.rate*z_differential_list
                self.rate *= 0.75
                print("continue")
                continue
            if abs(after_total_energy-self.current_total_energy) < 0.001:
                print("break")
                break
            self.current_total_energy = after_total_energy
        return after_total_energy

    # 中心差分を用いて微分値を出す
    cdef atom_formation_energy(self):
        cdef int i
        cdef double forward_occ, back_occ, k_bT, occ_i
        k_bT = self.BOLTZMANN_CONSTANT*self.temperature
        for i in range(self.atom_num):
            self.occupancy[i] += self.sigma
            forward_occ = self.culc_total_energy(i)
            self.occupancy[i] -= 2*self.sigma
            back_occ = self.culc_total_energy(i)
            self.occupancy[i] += self.sigma
            occ_i = self.occupancy[i]
            self.chem_differential_list[i] = (forward_occ-back_occ)/(2*self.sigma) - k_bT*(math.log(occ_i)-math.log(1-occ_i))    
        return self.chem_differential_list

    cdef atom_formation_inter(self, int i, int j):
        return <double>self.chem_differential_list[i] - <double>self.chem_differential_list[j]

    cdef jump_frequency(self, int i, int j):
        cdef double f_ij, gamma_ij, gamma_ji
        f_ij = self.atom_formation_inter(i, j)
        gamma_ij = self.frequency_factor * math.exp(-(self.activation_energy-f_ij*0.5)/(self.BOLTZMANN_CONSTANT*self.temperature))
        gamma_ji = self.frequency_factor * math.exp(-(self.activation_energy+f_ij*0.5)/(self.BOLTZMANN_CONSTANT*self.temperature))
        return gamma_ij, gamma_ji

    cdef update_concentration(self):
        cdef int i, j
        cdef double occ_i, occ_j, ret
        cdef np.ndarray[DOUBLE_t, ndim=1] change_concent
        change_concent = np.zeros(self.atom_num, dtype=float)
        for i in range(self.atom_num):
            ret = 0
            occ_i = self.occupancy[i]
            for j in range(self.atom_num):
                if i == j:
                    continue
                occ_j = self.occupancy[j]
                gamma_ij, gamma_ji = self.jump_frequency(i, j)
                ret += (1-occ_i)*occ_j*gamma_ji - (1-occ_j)*occ_i*gamma_ij
            change_concent[i] = ret
        self.occupancy += self.time_step * change_concent
        return change_concent

    cpdef main_loop(self):
        # 初期入力
        self.pos_init()
        ## while start
        # 自由エネルギー算出
        self.current_total_energy = self.culc_all_total_energy()
        print(self.current_total_energy)
        # エネルギーの最小化
        #print(self.update_info())
        # 微分値の生成
        # self.atom_formation_energy()
        # 濃度時間変化
        # print(self.update_concentration())
        ## while end
