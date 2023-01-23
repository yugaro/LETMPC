import numpy as np
from set_args import set_args
from model.vehicle import Vehicle
from model.gp import GP
from controller.mpc import MPC
np.random.seed(0)


if __name__ == '__main__':
    args = set_args()
    vehicle = Vehicle(args)
    z_train = np.load(args.datafile_z)
    y_train = np.load(args.datafile_y)

    gpmodels = GP(args, z_train, y_train)
    horizon = args.horizon
    mpcmodel = MPC(args, gpmodels, z_train, y_train, horizon)
    mpc = mpcmodel.setUp()
    # xreal_init = np.array([-2, -2, np.pi])
    xreal_init = np.array(
        [np.random.rand() - 2, np.random.rand() - 2, np.pi])
    mpcmodel.setInitial(mpc, xreal_init)

    xreal_traj = xreal_init
    xreal = xreal_init
    for t in range(args.step_max):
        u0, solver_stats = mpc.make_step(xreal)
        if solver_stats['success'] is False:
            print('Assumpition is not hold (MPC).')
            break
        state_list, input_list = mpcmodel.getStateInputList(mpc)
        xreal_next = vehicle.errRK4(xreal, input_list[0].reshape(-1))
        xreal = xreal_next
        xreal_traj = np.vstack([xreal_traj, xreal_next])
    np.save('../data/xreal_traj_periodic', xreal_traj)
