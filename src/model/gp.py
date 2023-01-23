import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
np.random.seed(0)


class GP:
    def __init__(self, args, z_train, y_train):
        self.z_train = z_train
        self.y_train = y_train
        self.noise = args.noise
        self.csk = [ConstantKernel() for i in range(y_train.shape[1])]
        self.rbfk = [RBF(length_scale=np.ones(z_train.shape[1]), length_scale_bounds=(1e-1, 1e10))
                     for i in range(y_train.shape[1])]
        self.whtk = [WhiteKernel(noise_level_bounds=(1e-10, 1e1))
                     for i in range(y_train.shape[1])]
        self.gpr = [GaussianProcessRegressor(
            alpha=0,
            kernel=self.csk[i] * self.rbfk[i] + self.whtk[i],
            normalize_y=True,
            random_state=0,
            n_restarts_optimizer=1
        ) for i in range(y_train.shape[1])]
        for i in range(y_train.shape[1]):
            self.gpr[i].fit(self.z_train, self.y_train[:, i])
        self.alpha = [np.sqrt(np.exp(self.gpr[i].kernel_.theta[0]))
                      for i in range(y_train.shape[1])]
        self.Lambda = [np.diag(np.exp(self.gpr[i].kernel_.theta[1: 1 + 5]) ** 2)
                       for i in range(y_train.shape[1])]
        self.Lambdax = [np.diag(np.exp(self.gpr[i].kernel_.theta[1: 1 + 3]) ** 2)
                        for i in range(y_train.shape[1])]
        self.noise = [np.sqrt(np.exp(self.gpr[i].kernel_.theta[-1]))
                      for i in range(y_train.shape[1])]

        self.b = np.array(args.b)
        self.cov = [self.gpr[i].L_ @ self.gpr[i].L_.T
                    for i in range(y_train.shape[1])]
        self.beta = np.array([self.betaF(i) for i in range(y_train.shape[1])])

        print(self.gpr[0].kernel_)
        print(self.gpr[1].kernel_)
        print(self.gpr[2].kernel_)
        print(self.beta)

    def betaF(self, i):
        betatmp = self.b[i] ** 2 - self.y_train[:, i] @ np.linalg.inv(
            self.cov[i]) @ self.y_train[:, i] + self.cov[i].shape[0]
        if betatmp < 0:
            return 1
        return np.sqrt(betatmp)

    def predict(self, z_test):
        means = np.zeros(self.y_train.shape[1])
        stds = np.zeros(self.y_train.shape[1])
        for i in range(self.y_train.shape[1]):
            means[i], stds[i] = self.gpr[i].predict(z_test, return_std=True)
        return means, stds
