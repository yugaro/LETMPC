import numpy as np
import do_mpc
from casadi import vertcat, SX
np.random.seed(0)


class MPC:
    def __init__(self, args, gpmodels, z_train, y_train, horizon):
        self.mpcmodel = do_mpc.model.Model(args.mpc_type)
        self.gpmodels = gpmodels
        self.alpha = gpmodels.alpha
        self.Lambda = gpmodels.Lambda
        self.cov = gpmodels.cov
        self.z_train = z_train
        self.y_train = y_train
        self.weightx = args.weightx
        self.horizon = horizon
        self.ts = args.ts
        self.setup_mpc = {
            'n_robust': 0,
            'n_horizon': horizon,
            't_step': args.ts,
            'state_discretization': args.mpc_type,
            'store_full_solution': True,
        }
        self.v_max = args.v_max
        self.omega_max = args.omega_max
        self.terminalset = args.terminalset

    def kernelF(self, i, distance):
        return (self.alpha[i] ** 2) * np.exp(-0.5 * distance.T @ np.linalg.inv(self.Lambda[i]) @ distance)

    def kstarF(self, i, zvar):
        kstar = SX.zeros(self.z_train.shape[0])
        for t in range(self.z_train.shape[0]):
            kstar[t] = self.kernelF(i, zvar - self.z_train[t, :])
        return kstar

    def muF(self, zvar):
        mu = [(self.kstarF(i, zvar)).T @ np.linalg.inv(self.cov[i]) @ self.y_train[:, i]
              for i in range(3)]
        return mu

    def setUp(self):
        xvar = self.mpcmodel.set_variable(
            var_type='_x', var_name='xvar', shape=(3, 1))
        uvar = self.mpcmodel.set_variable(
            var_type='_u', var_name='uvar', shape=(2, 1))
        zvar = vertcat(xvar, uvar)
        mu = self.muF(zvar)
        xvar_next = vertcat(xvar[0] + mu[0],
                            xvar[1] + mu[1],
                            xvar[2] + mu[2])
        self.mpcmodel.set_rhs(var_name='xvar', expr=xvar_next)
        costfunc = xvar.T @ np.diag(self.weightx) @ xvar
        self.mpcmodel.set_expression(expr_name='costfunc', expr=costfunc)
        self.mpcmodel.setup()

        mpc = do_mpc.controller.MPC(self.mpcmodel)
        mpc.set_param(**self.setup_mpc)
        lterm = self.mpcmodel.aux['costfunc']
        mterm = self.mpcmodel.aux['costfunc']
        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(uvar=1)
        mpc.bounds['lower', '_u', 'uvar'] = -np.array(
            [[self.v_max], [self.omega_max]])
        mpc.bounds['upper', '_u', 'uvar'] = np.array(
            [[self.v_max], [self.omega_max]])
        mpc.terminal_bounds['lower', '_x', 'xvar'] = -np.array(
            [[self.terminalset[0]], [self.terminalset[1]], [self.terminalset[2]]])
        mpc.terminal_bounds['upper', '_x', 'xvar'] = np.array(
            [[self.terminalset[0]], [self.terminalset[1]], [self.terminalset[2]]])
        mpc.setup()

        # simulator = do_mpc.simulator.Simulator(self.mpcmodel)
        # simulator.set_param(t_step=self.ts)
        # simulator.setup()

        return mpc

    def setInitial(self, mpc, x0):
        mpc.x0 = x0
        # simulator.x0 = x0
        mpc.set_initial_guess()

    def getStateInputList(self, mpc):
        state_list = [np.array(mpc.opt_x_num['_x', i, 0, 0]).reshape(-1)
                      for i in range(len(mpc.opt_x_num['_x', :, 0, 0]))]
        input_list = [np.array(mpc.opt_x_num['_u', i, 0]).reshape(-1)
                      for i in range(len(mpc.opt_x_num['_u', :, 0]))]
        return state_list, input_list
