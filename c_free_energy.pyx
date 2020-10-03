import math
import random
import numpy as np
cimport numpy as np

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
    cdef public list occupancy
    cdef public double[:] gauss_width
    cdef public list energy_list


    def __init__(self):
        self.mass = 1.0544429e-25
        self.rc = 5.50679
        self.h = 0.50037
        self.a = 3.80362
        self.b1 = 0.17394
        self.b2 = 5.25661*10**2
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
        self.lat_parameter = 4.05
        self.atom_num = 3**3
        self.x_pos = np.zeros(self.atom_num,dtype=float)
        self.y_pos = np.zeros(self.atom_num,dtype=float)
        self.z_pos = np.zeros(self.atom_num,dtype=float)
        self.gauss_width = np.zeros(self.atom_num,dtype=float)
        self.current_total_energy = 0
        self.sigma = 0.00001
        self.temperature = 10.0
        self.occupancy = [1.0] * self.atom_num
        self.energy_list = [0.0] * self.atom_num
        self.BOLTZMANN_CONSTANT = 1.380649e-23
        self.PI = math.pi
        self.DIRACS_CONSTANT = 1.054571817e-34

    # ドブロイ波長
    cdef thermal_wavelength(self):
        return <double>self.DIRACS_CONSTANT * ((2*self.PI)/(self.mass*self.BOLTZMANN_CONSTANT*self.temperature))**(0.5)

    cdef pos_init(self):
        cdef int i, x, y, z
        i = 0
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    self.x_pos[i] = self.lat_parameter*x
                    self.y_pos[i] = self.lat_parameter*y
                    self.z_pos[i] = self.lat_parameter*z
                    self.gauss_width[i] = 1.0
                    i += 1

    cdef two_atom_dis(self, int i, int j):
        cdef double dx, dy, dz
        dx = self.x_pos[i] - self.x_pos[j]
        dy = self.y_pos[i] - self.y_pos[j]
        dz = self.z_pos[i] - self.z_pos[j]
        return dx, dy, dz

    cdef abs_two_atom_dis(self, int i, int j):
        cdef double dx, dy, dz
        dx, dy, dz = self.two_atom_dis(i, j)
        return (dx**2+dy**2+dz**2)**0.5

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
        return <double>r**2+x**2+2*r*x*math.cos(theta)

    cdef change_func(self, function, double r, double X, double theta, double alpha):
        return <double>function(self.change_coo(r, X, theta))*(r**2)*2*math.sin(theta)*math.exp(-alpha*(r**2))

    # 積分
    cdef integral_sympson(self, function, int i, int j):
        cdef int n, m, i_r, j_theta
        cdef double dx, dy, s, X, alpha, x, x1, x2, sy1, sy2, sy3, y, y1, y2, sx
        n, m = 50, 50
        dx = self.rc/(2*n)
        dy = math.pi/(2*m)
        s = 0.0
        X = self.abs_two_atom_dis(i, j)
        alpha = self.alpha_int(i, j)
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

    cdef culc_rho(self, int i):
        cdef int j
        cdef double ret
        ret = 0.0
        for j in range(self.atom_num):
            if i == j:
                continue
            else:
                ret += <double>self.occupancy[j]*2*(self.alpha_int(i, j)**1.5)*self.integral_sympson(self.electron_density_function, i, j)/math.pi**0.5
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
            if i == j:
                continue
            pair += <double>self.occupancy[i]*self.occupancy[j]*2*(self.alpha_int(i, j)**1.5)*self.integral_sympson(self.pair_potential, i, j)/math.pi**0.5
        rho = self.culc_rho(i)
        embed = self.occupancy[i]*self.embedding_function(rho)
        vib_entropy = math.log(self.gauss_width[i]*(self.thermal_wavelength()**2)/math.pi)-1
        return pair*0.5 + embed + 1.5*self.BOLTZMANN_CONSTANT*self.temperature*vib_entropy
    
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
        cdef double rate = 0.001, after_total_energy
        cdef int i
        cdef np.ndarray[DOUBLE_t, ndim=1] gauss_differential_list, x_differential_list, y_differential_list, z_differential_list
        gauss_differential_list, x_differential_list, y_differential_list, z_differential_list = self.make_differential_list()
        self.gauss_width -= rate*gauss_differential_list
        self.x_pos -= rate*x_differential_list
        self.y_pos -= rate*y_differential_list
        self.z_pos -= rate*z_differential_list
        after_total_energy = self.culc_all_total_energy()
        return after_total_energy

    cpdef main_loop(self):
        self.pos_init()
        self.current_total_energy = self.culc_all_total_energy()
        print(self.current_total_energy)
        print(self.update_info())
