import numpy as np


class cubicSpline:
    def __init__(self, x, y):
        # Xが昇順になるようにソートする
        points = sorted(zip(x, y))
        self.x, self.y = np.array(list(zip(*points)), dtype=float)
        # フィッティングのために各係数を求めておく
        self.__initialize(self.x, self.y)

    def __initialize(self, x, y):
        xlen = len(x)  # 点の数
        N = xlen - 1  # 求めるべき変数の数（＝方程式の数）

        # Xが一致する値を持つ場合例外を発生させる
        if xlen != len(set(x)):
            raise ValueError("x must be different values")

        matrix = np.zeros([4*N, 4*N])
        Y = np.zeros([4*N])

        equation = 0
        for i in range(N):
            for j in range(4):
                matrix[equation, 4*i+j] = pow(x[i], j)
            Y[equation] = y[i]
            equation += 1
        for i in range(N):
            for j in range(4):
                matrix[equation, 4*i+j] = pow(x[i+1], j)
            Y[equation] = y[i+1]
            equation += 1
        for i in range(N-1):
            for j in range(4):
                matrix[equation, 4*i+j] = j*pow(x[i+1], j-1)
                matrix[equation, 4*(i+1)+j] = -j*pow(x[i+1], j-1)
            equation += 1
        for i in range(N-1):
            matrix[equation, 4*i+3] = 3*x[i+1]
            matrix[equation, 4*i+2] = 1
            matrix[equation, 4*(i+1)+3] = -3*x[i+1]
            matrix[equation, 4*(i+1)+2] = -1
            equation += 1
        matrix[equation, 3] = 3*x[0]
        matrix[equation, 2] = 1
        equation += 1
        matrix[4*N-1, 4*N-1] = 3*x[N]
        matrix[4*N-1, 4*N-2] = 1

        # Wa=Y => a=W^(-1)Yとして変数の行列を求める
        # その際、逆行列を求めるのにnp.linalg.invを使う
        self.variables = np.dot(np.linalg.inv(matrix), Y)
        print(self.variables)

    def fit(self, x):
        """
        引数xが該当する区間を見つけてきて補間後の値を返す
        """
        xlen = len(self.x)
        for index, j in enumerate(self.x):
            if x < j:
                index -= 1
                break
        if index == -1:
            index += 1
        elif index == xlen-1:
            index -= 1
        a3 = self.variables[4*index + 3]
        a2 = self.variables[4*index + 2]
        a1 = self.variables[4*index + 1]
        a0 = self.variables[4*index + 0]

        result = a3*pow(x, 3) + a2*pow(x, 2) + a1*x + a0
        return result
