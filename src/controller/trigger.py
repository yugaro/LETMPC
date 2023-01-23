import numpy as np
import cvxpy as cp
np.random.seed(0)


class Trigger:
    def __init__(self, args, gpmodels, state_list, input_list, horizon):
        self.b = np.array(args.b)
        self.gamma = np.array(args.gamma)
        self.horizon = horizon
        self.gpmodels = gpmodels
        self.alpha = gpmodels.alpha
        self.Lambdax = gpmodels.Lambdax
        self.beta = gpmodels.beta
        self.state_list = state_list
        self.input_list = input_list

    def deltaF(self, j):
        zsuc = np.hstack([self.state_list[j], self.input_list[j]]).reshape(1, -1)
        _, stdsuc = self.gpmodels.predict(zsuc)
        return self.beta * stdsuc

    def cF(self, i, psi_next):
        return np.sqrt(2 * np.log((2 * (self.alpha[i] ** 2)) / (2 * (self.alpha[i] ** 2) - (psi_next[i] ** 2))))

    def opt(self):
        xi_list = [-np.ones(len(self.alpha))]
        psi_next = self.gamma
        for j in reversed(range(self.horizon)):
            delta = self.deltaF(j)
            psi = cp.Variable(3, pos=True)
            constranits = [cp.quad_form(cp.multiply(self.b, psi), np.linalg.inv(self.Lambdax[i])) <= self.cF(i, psi_next) ** 2
                           for i in range(len(self.alpha))]
            constranits += [psi[i] <= np.sqrt(2) * self.alpha[i]
                            for i in range(len(self.alpha))]
            constranits += [delta[i] / self.b[i] <= psi[i]
                            for i in range(len(self.alpha))]

            # psi_func = cp.geo_mean(psi)
            # psi_func = psi[0] + psi[1] + psi[2]
            psi_func = cp.sum(psi)
            # psi_func = cp.sum(
            #     psi / np.diag(np.sqrt(self.Lambdax[0]))) + cp.sum(
            #         psi / np.diag(np.sqrt(self.Lambdax[1]))) + cp.sum(psi / np.diag(np.sqrt(self.Lambdax[2])))
            prob_psi = cp.Problem(cp.Maximize(psi_func), constranits)
            prob_psi.solve(solver=cp.MOSEK)

            print(prob_psi.status)
            if prob_psi.status == 'infeasible':
                return prob_psi.status, 0
            psi_next = psi.value
            # print(psi.value)
            xi_list.append(psi.value - delta / self.b)
        return prob_psi.status, list(reversed(xi_list))

    def kernelValue(self, i, distance):
        return (self.alpha[i] ** 2) * np.exp(-0.5 * distance @ np.linalg.inv(self.Lambdax[i]) @ distance)

    def kernelMetric(self, distance):
        return np.array([np.sqrt(2 * (self.alpha[i] ** 2) - 2 * self.kernelValue(i, distance)) for i in range(len(self.alpha))])

    def checkF(self, xpred, xreal, trigger_value):
        kmd = self.kernelMetric(xpred - xreal)
        # print('------------------')
        # print('xpred', xpred)
        # print('xreal', xreal)
        # print('kmd', kmd)
        # print('trigger_value', trigger_value)
        # print('------------------')
        return np.all(kmd <= trigger_value)

    def newDataCheck(self, x, u, xi1):
        z = np.hstack([x, u])
        _, std = self.gpmodels.predict(z.reshape(1, -1))
        # print('std:', std)
        # print('beta:', self.beta)
        # print('delta:', self.beta * std)
        # print('kmdstd:', self.kernelMetric(self.beta * std))
        # print('xi1:', xi1)
        return np.any(self.kernelMetric(self.beta * std) > xi1)
        # return np.all(self.kernelMetric(self.beta * std) <= xi1)
