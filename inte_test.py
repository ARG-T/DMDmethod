import random
import math


def main():

    # ∫(r=[0,1]) ∫(theta=[0,pi]) r^4*sin(theta) dr d(theta) = 2/5
    def integral_func(r, theta):
        return r**4 * math.sin(theta)

    def integral_monte():
        s = 0
        STEP = 100000
        for _ in range(STEP):
            r, theta = random.random(), random.random()*math.pi
            s += integral_func(r, theta)*math.pi
        return s / STEP

    def integral_sympson():
        n, m = 100, 100
        dx = 1/(2*n)
        dy = math.pi/(2*m)
        s = 0
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

                s1 = ((integral_func(x, y) + 4 * integral_func(x, y1) + integral_func(x, y2))/3)*dy
                s2 = ((integral_func(x1, y) + 4 * integral_func(x1, y1) + integral_func(x1, y2))/3)*dy
                s3 = ((integral_func(x2, y) + 4 * integral_func(x2, y1) + integral_func(x2, y2))/3)*dy

                sy1 += s1
                sy2 += s2
                sy3 += s3

            sx = ((sy1+4*sy2+sy3)/3)*dx
            s += sx

        return s

    def error_value(val):
        return 2/5 - val

    mo = integral_monte()
    sy = integral_sympson()

    print(mo, error_value(mo))
    print(sy, error_value(sy))


if __name__ == "__main__":
    main()
