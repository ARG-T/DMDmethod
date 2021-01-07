import numpy as np
cimport numpy as np
import time


ctypedef np.float64_t DOUBLE_t


cutoff = 5.5
lat_parameter_a = 2.951
lat_parameter_c = 4.6843
lat_parameter_x = 6*lat_parameter_a
lat_parameter_y = 3**(1.5)*lat_parameter_a
lat_parameter_z = 3*lat_parameter_c
Ti_atom_num = 216
O_atom_num = 1
phyO_X = np.array([1.0, 1.9, 2.8, 3.7, 4.6, 5.5, 6.4])
phyO_matrix = np.array([0.23881911, 0.155001356, 0, 0, 2.3236094436328854, -3.1345335685898332, 1.7291453854046557, -0.3027976125707842, -16.210338271364062, 16.723267554621188, -5.362926444313568, 0.5414966528718617, 12.855516907633962, -6.8436420499718, 1.006508583954807, -0.032326322647811545, 50.785629280514314, -31.580671858372032, 6.3841237596939875, -0.4220085817593464, 0, 0, 0, 0])
rhoO_X = np.array([1.0, 1.9, 2.8, 3.7, 4.6, 5.5, 6.4])
rhoO_matrix = np.array([128.996605, -61.98275852, 0, 0, -26.685016038551453, 183.8351303964215, -129.3825305040879, 22.6998932111252, 1190.0282939694275, -1119.7862731835557, 336.1965422030469, -32.72618687305753, -998.4886412935, 654.6869175701688, -143.3908066493111, 10.47988059111886, 110.74621246315735, -68.72711748852038, 13.873114015621326, -0.9160556889487035, 0, 0, 0, 0])
fO_X = np.array([1.0, 1.9, 2.8, 3.7, 4.6, 5.5, 6.4])
fO_matrix = np.array([-20.8088995, 12.33155479, 0, 0, 12.30617029625413, -39.91840370422211, 27.46381860444437, -4.8089421225071085, -175.56349280184324, 161.37052104373922, -44.425083091256134, 3.7492604603143818, -353.81094729209747, 305.8954841439455, -83.48588392914972, 7.268251526791282, 1304.7163051589664, -775.7527239763139, 151.65503087960235, -9.770945198480607, 0, 0, 0, 0])
UO_X = np.array([-32.0614825, -23.9928, -15.9241175, -7.855435, 0.2132475, 8.28193, 16.3506125])
UO_matrix = np.array([0.13523710625268848, -0.006767470050830504, -6.403932713977545e-10, -6.657971130799231e-12, -0.5565592211916941, -0.093267961128702, -0.003605269349250576, -5.008820963077018e-05, 0.10338926074052146, 0.03106203366043759, 0.004202384342650368, 0.00011334635624916939, 4.381108702818798e-10, -0.008422448218316628, -0.0008240059054713739, -9.994080302852572e-05, -1.3244440715595024e-06, -0.008403809563480327, -0.0009114097664137471, 3.6682699306201744e-05, 0.020836692233386828, -0.015952056171743827, 1.7775456310868687e-09, -3.62381048698299e-11])
gO_X = np.array([-1.0, -0.5, 1e-13, 0.5, 1.0])
gO_matrix = np.array([-0.628045699178449, -2.3126733423209807, -2.449864948713797, -0.8166216495712656, -0.2606508769999892, -0.10830440925022018, 1.9588729174277233, 2.1225369278564146, -0.2606508769999892, -0.10830440925022018, 1.958872917429236, -2.920631517857675, -0.8275693466786391, 3.293206408821675, -4.844148718714553, 1.6147162395715178])
phyTi_X = np.array([1.453676923, 1.7427, 2.031723077, 2.320746154, 2.609769231, 2.898792308, 3.187815385, 3.476838462, 3.765861538, 4.054884615, 4.343907692, 4.632930769, 4.921953846, 5.5, 5.789023077])
phyTi_matrix = np.array([38.5983, -20.0, 0, 0, 138.641016, -161.73799, 61.3580311, -7.44068525, 264.830161, -348.066253, 153.067509, -22.4869415, -97.1484485, 119.859041, -48.559581, 6.47315392, 32.9478747, -29.6901851, 8.74403977, -0.845963303, 51.9776738, -49.3843858, 15.5379728, -1.62720048, -25.3824356, 23.4179256, -7.2997084, 0.760817475, 54.099171, -45.1630057, 12.4253758, -1.13027644, -10.4366343, 6.24818244, -1.22653034, 0.0781151396, -24.6825804, 16.7880232, -3.8258252, 0.291791163, 0.814280939, -0.820680633, 0.227829952, -0.0192694029, 0.0674484399, -0.337078004, 0.123446214, -0.011759129, -5.27729441, 2.92061783, -0.538424241, 0.033065242, 0, 0, 0, 0])
rhoTi_X = np.array([1.82038, 2.0558, 2.29122, 2.52664, 2.76206, 2.99748, 3.2329, 3.46832, 3.70374, 3.93916, 4.17458, 4.41, 4.64542])
rhoTi_matrix = np.array([3.8033, -1, 0, 0, -3064.23873, 4263.58898, -1961.70616, 298.2907, 1250.22297, -1385.53467, 503.846541, -60.4050606, 1410.03373, -1575.2856, 578.946644, -70.3128305, 383.220459, -460.016625, 175.165055, -21.5833088, -1185.44742, 1109.97005, -348.603801, 36.662157, 1089.20826, -1000.81812, 304.304784, -30.6570286, -959.343518, 771.121998, -206.588131, 18.4438728, -275.020362, 216.825673, -56.9295901, 4.97473614, 89.3986807, -60.7099251, 13.5259381, -0.98723953, 368.205196, -261.070092, 61.5212256, -4.81958414, 0, 0, 0, 0])
fTi_X = np.array([1.794222222, 2.0558, 2.317377778, 2.578955556, 2.840533333, 3.102111111, 3.363688889, 3.625266667, 3.886844444, 4.148422222, 4.41, 4.671577778])
fTi_matrix = np.array([-5.84985, 2.7733, 0, 0, 448.74591441, -632.01326202, 293.44604076, -44.86373342, -259.76560021, 285.20211997, -102.35277577, 12.06826105, -293.77175671, 324.76017643, -117.6915653, 14.05081939, 53.25633892, -41.74996968, 11.33707233, -1.09054346, -52.67497511, 60.69444019, -21.68702197, 2.45801784, 105.41979458, -80.30680122, 20.23161143, -1.6960161, -25.77273211, 28.2583379, -9.71518948, 1.05751017, 207.10664981, -151.48596518, 36.52908545, -2.90836975, 347.74667126, -253.19211796, 61.04591314, -4.87834181, 0, 0, 0, 0])
UTi_X = np.array([-65.54363333, -55.1423, -44.74096667, -34.33963333, -23.9383, -13.53696667])
UTi_matrix = np.array([0.13265, 0.0078, 0, 0, -0.67329036, -0.0655428591, -0.00181122337, -1.35324663e-05, 4.21319602, 0.26210896, 0.0055120833, 4.10283178e-05, 5.61059238, 0.384189154, 0.00906716476, 7.55373395e-05, 3.09174, 0.1052, 0, 0])
gTi_X = np.array([-1.275485714, -1, -0.724514286, -0.449028571, -0.173542857, 0.101942857, 0.377428571, 0.652914286, 0.9284, 1.203885714])
gTi_matrix = np.array([8.4096, 8.3364, 0, 0, 26.6850027, 98.2759409, 120.345034, 48.677596, -3.43990851, -26.4624328, -51.8232392, -30.5332982, 0.365719628, -1.03669115, 4.80064454, 11.5010603, 0.307819187, -2.03760428, -0.966882564, 0.423052969, 0.296731016, -1.7112988, -4.16774905, 10.8892644, 3.09071723, -23.9193648, 54.6726985, -41.0768135, 18.3571745, -94.0654358, 162.108043, -95.9259447, 50.06858, -60.4025, 0, 0])


cdef extern from "<math.h>" nogil:
    double M_PI
    double exp(double x)
    double cos(double x)
    double sin(double x)
    double pow(double x, double y)

cdef class Culculation(object):

    cdef public int i
    cdef public double i_x_pos
    cdef public double i_y_pos
    cdef public double i_z_pos
    cdef public long[:] j_list
    cdef public size_t j_len
    cdef public double[:] x_pos
    cdef public double[:] y_pos
    cdef public double[:] z_pos
    cdef public double[:] occ
    cdef public double[:] alpha

    def __cinit__(self, int i, double i_x_pos, double i_y_pos, double i_z_pos, long[:] j_list, double[:] x_pos, double[:] y_pos, double[:] z_pos, double[:] occ, double[:] alpha):
        self.i = i
        print(i)
        self.i_x_pos = i_x_pos
        self.i_y_pos = i_y_pos
        self.i_z_pos = i_z_pos
        self.j_list = j_list
        self.j_len = len(self.j_list)
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.z_pos = z_pos
        self.occ = occ
        self.alpha = alpha   

    cdef double abs_two_atom_distance(self, int j):
        cdef double i_x, i_y, i_z, j_x, j_y, j_z
        cdef double dx, dy, dz
        i_x, i_y, i_z = self.i_x_pos, self.i_y_pos, self.i_z_pos
        j_x, j_y, j_z = self.x_pos[j], self.y_pos[j], self.z_pos[j]
        dx = abs(j_x - i_x)
        dx = min(dx, lat_parameter_x-dx)
        dy = abs(j_y - i_y)
        dy = min(dy, lat_parameter_y-dy)
        dz = abs(j_z - i_z)
        dz = min(dz, lat_parameter_z-dz)
        return dx**2 + dy**2 + dz**2

    # 大小比較
    cdef double judge_ls(self, double a, double b, double L):
        if a > b:
            if 2*(a-b) < L:
                return b-a
            else:
                return b+L-a
        else:
            if 2*(b-a) < L:
                return b-a
            else:
                return b-a-L

    # 3点のcosを算出する
    cdef double culc_cos(self, int j, int k):
        cdef double x_ij, y_ij, z_ij, x_ik, y_ik, z_ik, cos_theta
        x_ij = self.judge_ls(self.i_x_pos, self.x_pos[j], lat_parameter_x)
        y_ij = self.judge_ls(self.i_y_pos, self.y_pos[j], lat_parameter_y)
        z_ij = self.judge_ls(self.i_z_pos, self.z_pos[j], lat_parameter_z)
        x_ik = self.judge_ls(self.i_x_pos, self.x_pos[k], lat_parameter_x)
        y_ik = self.judge_ls(self.i_y_pos, self.y_pos[k], lat_parameter_y)
        z_ik = self.judge_ls(self.i_z_pos, self.z_pos[k], lat_parameter_z)
        cos_theta = (x_ij*x_ik+y_ij*y_ik+z_ij*z_ik)/(((x_ij**2+y_ij**2+z_ij**2)*(x_ik**2+y_ik**2+z_ik**2))**0.5)
        return cos_theta

    cdef double change_coo(self, double r, double X, double theta):
        return (r**2+X**2+r*X*cos(theta))**0.5

    cdef double ret_spline(self, double inp, double[:] matrix, double[:] x_data):
        """
        引数xが該当する区間を見つけてきて補間後の値を返す
        """
        cdef int index
        cdef size_t xlen
        cdef double a0, a1, a2, a3, result
        xlen = len(x_data)
        for index in range(xlen):
            j = x_data[index]
            if inp < j:
                index -= 1
                break
        if index == -1:
            index += 1
        elif index == xlen-1:
            index -= 1
        a3 = matrix[4*index + 3]
        a2 = matrix[4*index + 2]
        a1 = matrix[4*index + 1]
        a0 = matrix[4*index + 0]

        result = a3*pow(inp, 3) + a2*pow(inp, 2) + a1*inp + a0
        return result

    cdef double pair_potential(self, double coo_x, int j_index):
        if self.i < Ti_atom_num:
            if self.j_list[j_index] < Ti_atom_num:  # Ti-Ti
                return 2*self.ret_spline(coo_x, phyTi_matrix, phyTi_X)
            else:   #Ti-O
                return self.ret_spline(coo_x, phyO_matrix, phyO_X)
        else:
            if self.j_list[j_index] < Ti_atom_num:  #Ti-O
                return self.ret_spline(coo_x, phyO_matrix, phyO_X)
            else:   #O-O
                return 0

    cdef double embedding_function(self, double coo_x, int j_index):
        cdef int j = self.j_list[j_index], k_index
        cdef double ret, f_ij, f_ik, g_cos, coo_y
        if j < Ti_atom_num:  # j is Ti
            ret = self.ret_spline(coo_x, rhoTi_matrix, rhoTi_X)
            f_ij = self.ret_spline(coo_x, fTi_matrix, fTi_X)
        else:   # j is O
            ret = self.ret_spline(coo_x, rhoO_matrix, rhoO_X)
            f_ij = self.ret_spline(coo_x, fO_matrix, fO_X)
        for k_index in range(self.j_len):
            k = self.j_list[k_index]
            if k < j:
                if j >= Ti_atom_num and k >= Ti_atom_num:    # both j and k are O
                    f_ik = 0
                    g_cos = 0
                else:
                    coo_y = self.abs_two_atom_distance(k_index)**0.5
                    cos_ijk = self.culc_cos(j_index, k_index)
                    if j < Ti_atom_num and k < Ti_atom_num:     # both j and k are Ti
                        f_ik = self.ret_spline(coo_y, rhoTi_matrix, rhoTi_X)
                        g_cos = self.ret_spline(cos_ijk, gTi_matrix, gTi_X)
                    else:
                        f_ik = self.ret_spline(coo_y, rhoO_matrix, rhoO_X)
                        g_cos = self.ret_spline(cos_ijk, gO_matrix, gO_X)
        return ret + f_ij*f_ik*g_cos

    cdef double sympson(self, double[:,:] f, int m, int n, double h1, double h2):
        cdef np.ndarray[DOUBLE_t, ndim=1] temp
        cdef int _i, _j
        cdef double v
        temp = np.zeros(m+1)
        for _i in range(m+1):
            v = -f[_i, 0]+f[_i, n]
            for _j in range(0, n, 2):
                v += (2*f[_i, _j] + 4*f[_i, _j+1])
            temp[_i] = v
        v = -temp[0]+temp[m]
        for _i in range(0, m, 2):
            v += (2*temp[_i] + 4*temp[_i+1])
        return v*h1*h2/9


    cdef double culc_symp(self, int func_number, double X, int j_index, double alpha_int):
        cdef int m, n, _i, _j
        cdef double r0, r1, theta0, theta1, r, theta, h1, h2, s, coef
        cdef np.ndarray[DOUBLE_t, ndim=2] f
        m = 100
        n = 100
        r0, r1 = 0, cutoff
        theta0, theta1 = 0, M_PI
        f = np.zeros((m+1, n+1))
        for _i in range(m+1):
            r = r0+(r1-r0)/m*_i
            for _j in range(n+1):
                theta = theta0+(theta1-theta0)/n*_j
                if func_number == 0:
                    f[_i, _j] = self.pair_potential(self.change_coo(r, X, theta), j_index)*(r**2)*sin(theta)*exp(-alpha_int*(r**2))
                elif func_number == 1:
                    f[_i, _j] = self.embedding_function(self.change_coo(r, X, theta), j_index)*(r**2)*sin(theta)*exp(-alpha_int*(r**2))
        h1 = (r1-r0)/m
        h2 = (theta1-theta0)/n
        s = self.sympson(f, m, n, h1, h2)
        return s

    cpdef culc_data(self):
        cdef int index, j_index
        cdef double coef, alpha_int, pair_i = 0, base_rho = 0, X, embed_i
        cdef double s
        s = time.time()
        for j_index in range(self.j_len):
            alpha_int = self.alpha[j_index]
            occ_j = self.occ[j_index]
            X = self.abs_two_atom_distance(j_index)**0.5
            coef = occ_j*2*alpha_int**1.5/M_PI**0.5
            pair_i += coef*self.culc_symp(0, X, j_index, alpha_int)
            base_rho += coef*self.culc_symp(1, X, j_index, alpha_int)
        if self.i < Ti_atom_num:    # i is Ti
            embed_i = self.ret_spline(base_rho, UTi_matrix, UTi_X)
        else:
            embed_i = self.ret_spline(base_rho, UO_matrix, UO_X)
        print(pair_i/2, embed_i)
        return pair_i/2, embed_i
