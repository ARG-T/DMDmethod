import random
import numpy as np
cimport numpy as np
import csv
import time
import atom_info_input
from libc.math cimport sin, cos, log, exp, M_PI

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
    cdef public double[:] rho_list
    cdef public double[:] pair_list
    cdef public double[:] embed_list
    cdef public double[:] mixed_list
    cdef public double[:] vib_ent_list
    cdef public double[:] energy_list
    cdef public double[:] VG_energy_list
    cdef public list reference_neighbor_list
    cdef public list Verlet_neighbor_list

    def __cinit__(self, lat_parameter, temperature):
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
        self.lat_parameter = lat_parameter
        self.atom_num = 108
        self.x_pos = np.zeros(self.atom_num,dtype=float)
        self.y_pos = np.zeros(self.atom_num,dtype=float)
        self.z_pos = np.zeros(self.atom_num,dtype=float)
        self.gauss_width = np.zeros(self.atom_num,dtype=float)
        self.current_total_energy = 0
        self.sigma = 1e-10
        self.temperature = temperature
        self.rho_list = np.zeros(self.atom_num, dtype=float)
        self.pair_list = np.zeros(self.atom_num, dtype=float)
        self.embed_list = np.zeros(self.atom_num, dtype=float)
        self.mixed_list = np.zeros(self.atom_num, dtype=float)
        self.vib_ent_list = np.zeros(self.atom_num, dtype=float)
        self.occupancy = np.zeros(self.atom_num,dtype=float)
        self.energy_list = np.zeros(self.atom_num, dtype=float)
        self.VG_energy_list = np.zeros(self.atom_num, dtype=float)
        self.BOLTZMANN_CONSTANT = 8.6173336e-5
        self.PI = M_PI
        self.DIRACS_CONSTANT = 6.5821198e-16
        self.reference_neighbor_list = [0]*(self.atom_num**2)
        self.Verlet_neighbor_list = [0]*(self.atom_num+1)
        # TODO:初期値
        self.activation_energy = 2.041761       # eV
        # TODO:初期値
        self.frequency_factor = 3.58128*10**14
        self.time_step = 0.000001
        self.rate = 0.1

    # ドブロイ波長
    cdef thermal_wavelength(self):
        cdef double planck, boltzman, ret
        planck = 6.62607015E-34
        boltzman = 1.380649E-23
        ret = 10**10*planck/((2*self.PI*self.mass*boltzman*self.temperature)**0.5)
        return ret

    # 初期配置入力
    cdef pos_init(self):
        cdef int i
        cdef list info, row
        with open("atom_info.csv") as f:
            reader = csv.reader(f)
            info = [row for row in reader]

        for i in range(self.atom_num):
            self.x_pos[i] = float(info[i][1])
            self.y_pos[i] = float(info[i][2])
            self.z_pos[i] = float(info[i][3])
            self.occupancy[i] = float(info[i][4])
            self.gauss_width[i] = float(info[i][5])

    # 同等の位置を考慮しつつ二つの原子の距離(A)を返す
    cdef abs_two_atom_distance(self, int i, int j):
        cdef double x_pos_i, y_pos_i, z_pos_i, x_pos_j, y_pos_j, z_pos_j
        cdef double dx, dy, dz, dr2
        x_pos_i, y_pos_i, z_pos_i = self.x_pos[i], self.y_pos[i], self.z_pos[i]
        x_pos_j, y_pos_j, z_pos_j = self.x_pos[j], self.y_pos[j], self.z_pos[j]
        dx = abs(x_pos_i - x_pos_j)
        dx = min(dx, self.lat_parameter*3-dx)
        dy = abs(y_pos_i - y_pos_j)
        dy = min(dy, self.lat_parameter*3-dy)
        dz = abs(z_pos_i - z_pos_j)
        dz = min(dz, self.lat_parameter*3-dz)
        dr2 = dx**2+dy**2+dz**2
        return dr2

    # 近接リスト作成
    # 周期境界条件(~period)
    cdef make_Verlet_neighbor_list(self):
        cdef int nlist, i, j
        cdef double dr2
        nlist = -1      # 0-indexed
        for i in range(self.atom_num):
            self.Verlet_neighbor_list[i] = nlist+1
            for j in range(self.atom_num):
                if i == j:
                    continue
                dr2 = self.abs_two_atom_distance(i, j)
                if dr2 <= (self.rc)**2:
                    nlist += 1
                    self.reference_neighbor_list[nlist] = j
        self.Verlet_neighbor_list[self.atom_num] = nlist+1

    cdef Morse_function(self, double r, double r_0, double a):
        cdef double exp_val, ret
        exp_val = exp(-a*(r-r_0))
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
        func1 = self.a*exp(-self.b1*(r-self.r0_list_aft[0])**2)
        func2 = exp(-self.b2*(r-self.r0_list_aft[1]))
        return (func1+func2)*self.cutoff_function(<double>(r-self.rc)/self.h)

    cdef change_coo(self, double r, double x, double theta):
        return <double>(r**2+x**2+2*r*x*cos(theta))**0.5

    cdef change_func(self, function, double r, double X, double theta, double alpha):
        return <double>function(self.change_coo(r, X, theta))*(r**2)*sin(theta)*exp(-alpha*(r**2))

    # 積分
    cdef integral_sympson(self, function, int i, int j):
        cdef int n, m, i_r, j_theta
        cdef double dx, dy, s, X, alpha, x, x1, x2, sy1, sy2, sy3, y, y1, y2, sx
        n, m = 55*2, 31*2
        dx = self.rc/(2*n)
        dy = self.PI/(2*m)
        s = 0
        X = self.abs_two_atom_distance(i, j)**0.5
        alpha = self.alpha_int(i, j)
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

    cdef embedding_function(self, double rho):
        cdef int i
        cdef double ret, qn
        ret = 0.0
        if rho <= 1:
            ret += self.F0 + (self.F2*(rho-1)**2)*0.5
            for i, qn in enumerate(self.qn_list):
                ret += qn*(rho-1)**(i+3)
        else:
            ret += (self.F0 + (self.F2*(rho-1)**2)*0.5 + self.qn_list[0]*(rho-1)**3 + self.Q1*(rho-1)**4)/(1+self.Q2*(rho-1)**3)
        return ret

    cdef mixed_entropy(self, int i):
        cdef double occ_i
        occ_i = self.occupancy[i]
        return <double>occ_i*log(occ_i) + (1-occ_i)*log(1-occ_i)

    cdef culc_VG_energy(self, int i):
        cdef int start, end
        cdef list j_list
        cdef int j
        cdef double base_pair, base_rho, occ_i, occ_j, occ_inter, pair_i, rho_i, embed_i, vib_ent, VG_energy_i, alpha_int
        occ_i = self.occupancy[i]
        vib_ent = log((self.gauss_width[i]*self.thermal_wavelength()**2)/self.PI)-1
        start = self.Verlet_neighbor_list[i]
        end = self.Verlet_neighbor_list[i+1]
        # 条件に合わないものはここで弾く
        if start > end:
            return 0
        pair_i = 0
        rho_i = 0
        j_list = self.reference_neighbor_list[start:end]
        for j in j_list:
            occ_j = self.occupancy[j]
            alpha_int = self.alpha_int(i, j)
            base_pair = 2*self.PI*((alpha_int/self.PI)**1.5)*self.integral_sympson(self.pair_potential, i, j)
            pair_i += occ_j*base_pair
            base_rho = 2*self.PI*((alpha_int/self.PI)**1.5)*self.integral_sympson(self.electron_density_function, i, j)
            rho_i += occ_j*base_rho
        embed_i = self.embedding_function(rho_i)
        # VG_energyの算出
        VG_energy_i = 0.5*pair_i + embed_i + 1.5*self.BOLTZMANN_CONSTANT*self.temperature*vib_ent
        return VG_energy_i
  
    cdef culc_all_total_energy(self):
        cdef double VG_energy, mixed_energy
        cdef int i
        for i in range(self.atom_num):
            self.VG_energy_list[i] = self.occupancy[i]*self.culc_VG_energy(i)
            # mixed_ent_listの作成
            self.mixed_list[i] = self.mixed_entropy(i)
        VG_energy = np.sum(self.VG_energy_list)
        # mixed_entropy_listの作成と総和
        mixed_energy = self.BOLTZMANN_CONSTANT*self.temperature*np.sum(self.mixed_list)
        return VG_energy + mixed_energy

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
            gauss_differential_list[i] = (forward_gauss-back_gauss)/self.sigma
            # baseに戻す
            self.gauss_width[i] += self.sigma
            # 位置(x)
            self.x_pos[i] += self.sigma
            forward_x = self.culc_VG_energy(i)
            self.x_pos[i] -= 2*self.sigma
            back_x = self.culc_VG_energy(i)
            x_differential_list[i] = (forward_x-back_x)/self.sigma
            self.x_pos[i] += self.sigma
            # 位置(y)
            self.y_pos[i] += self.sigma
            forward_y = self.culc_VG_energy(i)
            self.y_pos[i] -= 2*self.sigma
            back_y = self.culc_VG_energy(i)
            y_differential_list[i] = (forward_y-back_y)/self.sigma
            self.y_pos[i] += self.sigma
            # 位置(z)
            self.z_pos[i] += self.sigma
            forward_z = self.culc_VG_energy(i)
            self.z_pos[i] -= 2*self.sigma
            back_z = self.culc_VG_energy(i)
            z_differential_list[i] = (forward_z-back_z)/self.sigma
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
            print(after_total_energy/self.atom_num)
            if abs(after_total_energy-self.current_total_energy) < 0.001:
                print("break")
                break
            elif after_total_energy > self.current_total_energy:
                self.gauss_width += self.rate*gauss_differential_list
                self.x_pos += self.rate*x_differential_list
                self.y_pos += self.rate*y_differential_list
                self.z_pos += self.rate*z_differential_list
                self.rate *= 0.75
                print("continue")
                continue
            self.current_total_energy = after_total_energy

    cdef atom_formation_inter(self, int i, int j):
        return self.VG_energy_list[i] - self.VG_energy_list[j]

    cdef jump_frequency(self, int i, int j):
        cdef double f_ij, gamma_ij, gamma_ji
        f_ij = self.atom_formation_inter(i, j)
        gamma_ij = self.frequency_factor * exp(-(self.activation_energy-f_ij*0.5)/(self.BOLTZMANN_CONSTANT*self.temperature))
        gamma_ji = self.frequency_factor * exp(-(self.activation_energy+f_ij*0.5)/(self.BOLTZMANN_CONSTANT*self.temperature))
        return gamma_ij, gamma_ji

    cdef update_concentration(self):
        cdef int i, j
        cdef double occ_i, occ_j, ret
        cdef np.ndarray[DOUBLE_t, ndim=1] change_concent
        change_concent = np.zeros(self.atom_num, dtype=float)
        for i in range(self.atom_num):
            start = self.Verlet_neighbor_list[i]
            end = self.Verlet_neighbor_list[i+1]
            # 条件に合わないものはここで弾く
            if start > end:
                return 0
            ret = 0
            occ_i = self.occupancy[i]
            j_list = self.reference_neighbor_list[start:end]
            for j in j_list:
                occ_j = self.occupancy[j]
                gamma_ij, gamma_ji = self.jump_frequency(i, j)
                ret += (1-occ_i)*occ_j*gamma_ji - (1-occ_j)*occ_i*gamma_ij
            change_concent[i] = ret
        self.occupancy += self.time_step*change_concent

    cpdef main_loop(self):
        atom_info_input.pos_input(self.lat_parameter, self.temperature)
        # 初期入力
        self.pos_init()
        self.make_Verlet_neighbor_list()
        ## while start
        # while True:
        # 自由エネルギー算出
        self.current_total_energy = self.culc_all_total_energy()
        print("lat parameter is ", self.lat_parameter, "free_energy/atom is ", self.current_total_energy/self.atom_num)
        return self.current_total_energy/self.atom_num
        # エネルギーの最小化
        # self.update_info()
        # 濃度時間変化
        # self.update_concentration()
        ## while end
