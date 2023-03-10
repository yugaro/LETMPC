import numpy as np
np.random.seed(0)


class Vehicle:
    def __init__(self, args):
        self.args = args
        self.ts = args.ts
        self.noise = args.noise
        self.v_r = args.v_r
        self.omega_r = args.omega_r
        self.Kx = args.Kx
        self.Ky = args.Ky
        self.Ktheta = args.Ktheta

    def getRF(self, x, u):
        f0 = np.cos(x[2]) * u[0]
        f1 = np.sin(x[2]) * u[0]
        f2 = u[1]
        return np.array([f0, f1, f2])

    def realRK4(self, x, u):
        k1 = self.getRF(x, u)
        k2 = self.getRF(x + self.ts / 2 * k1[2], u)
        k3 = self.getRF(x + self.ts / 2 * k2[2], u)
        k4 = self.getRF(x + self.ts * k3[2], u)
        x_next = x + self.ts / 6 * \
            (k1 + 2 * k2 + 2 * k3 + k4)
        return x_next

    def getEF(self, x, u):
        f0 = u[1] * x[1] - u[0] + self.v_r * np.cos(x[2])
        f1 = -u[1] * x[0] + self.v_r * np.sin(x[2])
        f2 = self.omega_r - u[1]
        return np.array([f0, f1, f2])

    def errRK4(self, x, u):
        k1 = self.getEF(x, u)
        k2 = self.getEF(x + self.ts / 2 * k1[2], u)
        k3 = self.getEF(x + self.ts / 2 * k2[2], u)
        k4 = self.getEF(x + self.ts * k3[2], u)
        x_next = x + self.ts / 6 * \
            (k1 + 2 * k2 + 2 * k3 + k4) + 2 * \
            self.noise * np.random.rand(3) - self.noise
        return x_next

    def getPIDCon(self, x, Kx, Ky, Ktheta):
        v = self.v_r * np.cos(x[2]) + Kx * x[0]
        omega = self.omega_r + self.v_r * Ky * \
            x[1] + Ktheta * np.sin(x[2])
        return np.array([v, omega])

    def getPIDConRand(self, x):
        v = self.args.v_max + 1e5
        omega = self.args.omega_max + 1e5
        while v > self.args.v_max or omega > self.args.omega_max:
            Kx = np.random.rand()
            Ky = np.random.rand()
            Ktheta = np.random.rand()
            v = self.v_r * np.cos(x[2]) + Kx * x[0]
            omega = self.omega_r + self.v_r * Ky * \
                x[1] + Ktheta * np.sin(x[2])
        return np.array([v, omega])
