import numpy as np
cimport numpy as np
import csv
import atom_info_input
from cython.parallel cimport prange
from cython cimport boundscheck, wraparound


ctypedef np.float64_t DOUBLE_t

cdef extern from "<math.h>" nogil:
    DEF DIRACS_CONSTANT = 6.5821198e-16
    DEF PLANCK_CONSTANT = 6.62607015E-34
    DEF BOLTZMANN_CONSTANT = 1.380649E-23
    DEF BOLTZMANN_CONSTANT_EV = 8.6173336e-5
    double M_PI
    double sin(double x)
    double cos(double x)
    double log(double x)
    double exp(double x)

cdef class FreeEnergy(object):

    cdef double mass
    cdef double rc
    cdef double h
    cdef double a
    cdef double b1
    cdef double b2
    cdef double F0
    cdef double F2
    cdef double Q1
    cdef double Q2
    cdef public double lat_parameter
    cdef public double temperature
    cdef double sigma
    cdef double current_total_energy
    cdef double activation_energy
    cdef double frequency_factor
    cdef double time_step
    cdef double rate
    cdef list qn_list
    cdef int atom_num
    cdef double[:] x_pos
    cdef double[:] y_pos
    cdef double[:] z_pos
    cdef double[:] occupancy
    cdef double[:] gauss_width
    cdef double[:] rho_list
    cdef double[:] pair_list
    cdef double[:] embed_list
    cdef double[:] mixed_list
    cdef double[:] vib_ent_list
    cdef double[:] energy_list
    cdef double[:] VG_energy_list
    cdef list reference_neighbor_list
    cdef list Verlet_neighbor_list

    def __cinit__(self, lat_parameter, temperature):
        self.mass = 1.0544429e-25
        self.rc = 5.50679
        self.h = 0.50037
        self.a = 3.80362
        self.b1 = 0.17394
        self.b2 = 5.35661*10**2
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
        self.reference_neighbor_list = [0]*(self.atom_num**2)
        self.Verlet_neighbor_list = [0]*(self.atom_num+1)

    # ドブロイ波長
    cdef thermal_wavelength(self):
        cdef double ret
        ret = 10**10*PLANCK_CONSTANT/((2*M_PI*self.mass*BOLTZMANN_CONSTANT*self.temperature)**0.5)
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
    cdef double abs_two_atom_distance(self, int i, int j):
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

    # 3点のcosを算出する
    cdef culc_cos(self, int i, int j, int k):
        x_pos_i, y_pos_i, z_pos_i = self.x_pos[i], self.y_pos[i], self.z_pos[i]
        x_pos_j, y_pos_j, z_pos_j = self.x_pos[j], self.y_pos[j], self.z_pos[j]
        x_pos_k, y_pos_k, z_pos_k = self.x_pos[k], self.y_pos[k], self.z_pos[k]
        x_ij = x_pos_j-x_pos_i
        x_ij = min(x_ij)



    # 近接リスト作成
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

    cdef double Morse_function(self, double r, double r_0, double a) nogil:
        cdef double exp_val, ret
        exp_val = exp(-a*(r-r_0))
        ret = exp_val**2-2*exp_val
        return ret

    cdef double cutoff_function(self, double x) nogil:
        if x >= 0:
            return 0.0
        else:
            return x**4/(1+x**4)

    cdef double alpha_int(self, int i, int j):
        cdef double o_i, o_j
        o_i = self.gauss_width[i]
        o_j = self.gauss_width[j]
        return o_i*o_j/(o_i+o_j)

    # 二体間ポテンシャル 
    # TODO: phyをspline補間から呼び出せるようにする
    cdef double pair_potential(self, double r) nogil:
        cdef:
            double ret
            double delta = 0.0086225
            double E1 = 2.01458*10**2
            double E2 = 0.00659288
            double alpha1 = 2.97758
            double alpha2 = 1.54927
            double r01 = 0.83591
            double r02 = 4.46867
            double rs1 = 2.24
            double rs2 = 1.80
            double rs3 = 1.20
            double S1 = 4.00
            double S2 = 40.00
            double S3 = 1.15*10**3
        ret = delta
        ret += E1*self.Morse_function(r, r01, alpha1)
        ret += E2*self.Morse_function(r, r02, alpha2)
        ret *= self.cutoff_function(<double>(r-self.rc)/self.h)
        if rs1 >= r:
            ret -= S1*(rs1-r)**4
        if rs2 >= r:
            ret -= S2*(rs2-r)**4
        if rs3 >= r:
            ret -= S2*(rs2-r)**4
        return ret

    # 電子密度関数
    # TODO: rhoをスプライン補間から呼び出せるようにする
    cdef double electron_density_function(self, double r, int j) nogil:
        cdef:
            int start, end, k
            double ret, mo_ret
        start = self.Verlet_neighbor_list[i]
        end = self.Verlet_neighbor_list[i+1]
        j_list = self.reference_neighbor_list[start:end]
        """
        ret = rho(r)
        """ 
        mo_ret = 0
        for k in j_list:
            if k == j:
                continue
            """
            mo_ret += 
            """
        return 

    # modified part
    # TODO: f, gをスプライン補間から呼び出せるようにする
    cdef double modified_EAM(self) nogil:
        return ret

    cdef double change_coo(self, double r, double x, double theta) nogil:
        return <double>(r**2+x**2+2*r*x*cos(theta))**0.5

    cdef double change_func_pair(self, double r, double X, double theta, double alpha) nogil:
        return self.pair_potential(self.change_coo(r, X, theta))*(r**2)*sin(theta)*exp(-alpha*(r**2))

    cdef double change_func_elec(self, double r, double X, double theta, double alpha, int j) nogil:
        return self.electron_density_function(self.change_coo(r, X, theta), j)*(r**2)*sin(theta)*exp(-alpha*(r**2))

    # 積分
    # TODO: 並列化
    # @boundscheck(False)
    # @wraparound(False)
    cdef double integral_sympson_pair(self, int i, int j):
        cdef int n, m, i_r, j_theta
        cdef double dx, dy, s, X, alpha, x, x1, x2, sy1, sy2, sy3, y, y1, y2, sx, s1, s2, s3
        n, m = 55*2, 31*2
        dx = self.rc/(2*n)
        dy = M_PI/(2*m)
        s = 0
        X = self.abs_two_atom_distance(i, j)**0.5
        alpha = self.alpha_int(i, j)
        for i_r in prange(n, nogil=True):
            x = 2*i_r*dx
            x1 = x+dx
            x2 = x+2*dx
            sy1, sy2, sy3 = 0, 0, 0
            # theta
            for j_theta in range(m):
                y = 2*j_theta*dy
                y1 = y+dy
                y2 = y+2*dy

                s1 = (self.change_func_pair(x, X, y, alpha) + 4 * self.change_func_pair(x, X, y1, alpha) + self.change_func_pair(x, X, y2, alpha))*dy/3
                s2 = (self.change_func_pair(x1, X, y, alpha) + 4 * self.change_func_pair(x1, X, y1, alpha) + self.change_func_pair(x1, X, y2, alpha))*dy/3
                s3 = (self.change_func_pair(x2, X, y, alpha) + 4 * self.change_func_pair(x2, X, y1, alpha) + self.change_func_pair(x2, X, y2, alpha))*dy/3

                sy1 = sy1+s1
                sy2 = sy2+s2
                sy3 = sy3+s3

            sx = (sy1+4*sy2+sy3)*dx/3
            s += sx

        return s

    # 積分
    # TODO: 並列化
    # @boundscheck(False)
    # @wraparound(False)
    cdef double integral_sympson_elec(self, int i, int j):
        cdef int n, m, i_r, j_theta
        cdef double dx, dy, s, X, alpha, x, x1, x2, sy1, sy2, sy3, y, y1, y2, sx, s1, s2, s3
        n, m = 55*2, 31*2
        dx = self.rc/(2*n)
        dy = M_PI/(2*m)
        s = 0
        X = self.abs_two_atom_distance(i, j)**0.5
        alpha = self.alpha_int(i, j)
        for i_r in prange(n, nogil=True):
            x = 2*i_r*dx
            x1 = x+dx
            x2 = x+2*dx
            sy1, sy2, sy3 = 0, 0, 0
            # theta
            for j_theta in range(m):
                y = 2*j_theta*dy
                y1 = y+dy
                y2 = y+2*dy

                s1 = (self.change_func_elec(x, X, y, alpha, j) + 4 * self.change_func_elec(x, X, y1, alpha, j) + self.change_func_elec(x, X, y2, alpha, j))*dy/3
                s2 = (self.change_func_elec(x1, X, y, alpha, j) + 4 * self.change_func_elec(x1, X, y1, alpha, j) + self.change_func_elec(x1, X, y2, alpha, j))*dy/3
                s3 = (self.change_func_elec(x2, X, y, alpha, j) + 4 * self.change_func_elec(x2, X, y1, alpha, j) + self.change_func_elec(x2, X, y2, alpha, j))*dy/3

                sy1 = sy1 + s1
                sy2 = sy2 + s2
                sy3 = sy3 + s3

            sx = (sy1+4*sy2+sy3)*dx/3
            s += sx

        return s

    # embedding function
    # TODO: Uをスプライン補間から呼び出せるようにする
    cdef embedding_function(self, double rho):
        cdef double ret
        """
        ret = f(rho)
        """
        return ret

    cdef double mixed_entropy(self, int i):
        cdef double occ_i
        occ_i = self.occupancy[i]
        return occ_i*log(occ_i) + (1-occ_i)*log(1-occ_i)

    cdef double culc_VG_energy(self, int i):
        cdef int start, end
        cdef list j_list
        cdef int j
        cdef double base_pair, base_rho, occ_i, occ_j, occ_inter, pair_i, rho_i, embed_i, vib_ent, VG_energy_i, alpha_int
        occ_i = self.occupancy[i]
        vib_ent = log((self.gauss_width[i]*self.thermal_wavelength()**2)/M_PI)-1
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
            base_pair = 2*M_PI*((alpha_int/M_PI)**1.5)*self.integral_sympson_pair(i, j)
            pair_i += occ_j*base_pair
            base_rho = 2*M_PI*((alpha_int/M_PI)**1.5)*self.integral_sympson_elec(i, j)
            rho_i += occ_j*base_rho
        embed_i = self.embedding_function(rho_i)
        # VG_energyの算出
        VG_energy_i = 0.5*pair_i + embed_i + 1.5*BOLTZMANN_CONSTANT_EV*self.temperature*vib_ent
        return VG_energy_i

    cdef double culc_all_total_energy(self):
        cdef double VG_energy, mixed_energy
        cdef int i
        for i in range(self.atom_num):
            self.VG_energy_list[i] = self.occupancy[i]*self.culc_VG_energy(i)
            # mixed_ent_listの作成
            self.mixed_list[i] = self.mixed_entropy(i)
        VG_energy = np.sum(self.VG_energy_list)
        # mixed_entropy_listの作成と総和
        mixed_energy = BOLTZMANN_CONSTANT_EV*self.temperature*np.sum(self.mixed_list)
        return VG_energy + mixed_energy

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
