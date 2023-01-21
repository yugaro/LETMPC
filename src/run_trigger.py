import numpy as np
from set_args import set_args
from model.vehicle import Vehicle
from model.gp import GP
from controller.mpc import MPC
from controller.trigger import Trigger
import time
np.random.seed(0)


# def iterTask(self, args, vehicle, z_train, y_train):
#     gpmodels = GP(args, z_train, y_train)


if __name__ == '__main__':
    args = set_args()
    vehicle = Vehicle(args)
    z_train = np.load(args.datafile_z)
    y_train = np.load(args.datafile_y)

    for iter in range(2):
        # show data
        print('iter:', iter)
        print('number of data points:', z_train.shape[0])
        # define GP model
        gpmodels = GP(args, z_train, y_train)

        # define horizon of MPC
        horizon = args.horizon

        # give initial state
        # thetainit = np.pi * np.random.rand()
        # rotationinit = np.array(
        #     [[np.cos(-thetainit), np.sin(-thetainit)],
        #      [-np.sin(-thetainit), np.cos(-thetainit)]])
        # posinit = rotationinit @ (np.random.rand(2) + 2)
        # xinit = np.array([posinit[0], posinit[1], thetainit])
        # xinit = np.array([np.random.rand() - 3, np.random.rand() - 3, np.pi * np.random.rand()])
        xinit = np.array(
            [np.random.rand() - 3, np.random.rand() - 3, np.pi])

        # set initial state and step
        xreal = xinit
        xreal_traj = xinit
        step = 0

        # define new data box
        z_train_new = np.zeros((1, 5))
        y_train_new = np.zeros((1, 3))

        # execute mpc until the horizon becomes 1 or the state enters into the terminal set
        while (horizon != 1) and (np.any(np.abs(xreal) >= np.array(args.terminalset))):
            # define MPC
            mpcStartTime = time.time()
            mpcmodel = MPC(args, gpmodels, z_train, y_train, horizon)
            mpc = mpcmodel.setUp()
            mpcmodel.setInitial(mpc, xreal)

            # obtain control input
            u, mpc_status = mpc.make_step(xreal)
            mpcEndTime = time.time()
            print('MPC status:', mpc_status['success'])
            print('MPC time:', mpcEndTime - mpcStartTime)
            if mpc_status['success'] is False:
                print('Assumption is not hold. (MPC)')
                # reset MPC
                mpc.reset_history()
                break
            xpred_list, input_list = mpcmodel.getStateInputList(mpc)

            # obtain params of trigger
            trigger = Trigger(args, gpmodels, xpred_list, input_list, horizon)

            triggerStartTime = time.time()
            prob_status, xi_list = trigger.opt()
            triggerEndTime = time.time()
            print('Trigger time:', triggerEndTime - triggerStartTime)

            if prob_status == 'optimal':
                for j in range(horizon + 1):
                    # check event-trigger conditions
                    if trigger.checkF(xpred_list[j], xreal, xi_list[j]):
                        # add control input
                        xreal_next = vehicle.errRK4(xreal, input_list[j].reshape(-1))

                        # add new data point
                        if trigger.newDataCheck(xreal, input_list[j], xi_list[-2]):
                            uPID = vehicle.getPIDConRand(xreal)
                            z_point_new = np.hstack([xreal, uPID])
                            z_train_new = np.vstack([z_train_new, z_point_new])
                            y_train_new = np.vstack([y_train_new, xreal_next - xreal])

                        # accumulate state trajectory data
                        xreal_traj = np.vstack([xreal_traj, xreal_next])

                        # update state
                        xreal = xreal_next
                    else:
                        # decrease horizon
                        horizon -= j - 1

                        # reset MPC
                        mpc.reset_history()
                        # simulator.reset_history()

                        print('trigger:', j)
                        print('new horizon:', horizon)
                        break

                    # increase step
                    step += 1
            else:
                for j in range(horizon):
                    # get PID control input
                    uPID = vehicle.getPIDConRand(xreal)

                    # add control input
                    xreal_next = vehicle.errRK4(xreal, uPID.reshape(-1))

                    # add new data point
                    z_point_new = np.hstack([xreal, uPID])
                    z_train_new = np.vstack([z_train_new, z_point_new])
                    y_train_new = np.vstack([y_train_new, xreal_next - xreal])

                    # accumulate state trajectory data
                    xreal_traj = np.vstack([xreal_traj, xreal_next])

                    # update state
                    xreal = xreal_next

                    # increase step
                    step += 1

                # reset MPC
                mpc.reset_history()
                # simulator.reset_history()
                break
        # update dataset
        z_train = np.vstack([z_train, z_train_new[1:, :]])
        y_train = np.vstack([y_train, y_train_new[1:, :]])

        print(z_train_new[1:, :])
        print(z_train)

        for i in range(z_train_new[1:, :].shape[0]):
            x = z_train_new[i + 1, :3]
            u = vehicle.getPIDCon(x, args.Kx, args.Ky, args.Ktheta)
            # print('----------------------------------')
            # print('x:', x)
            # print('u:', u)
            # print('mpc:', z_train_new[i + 1, 3: 5])
            # print('----------------------------------')
        # print(y_train_new[1:, :])
        # print(y_train)

        # save trajectory data
        np.save('../data/xreal_traj_trigger_iter{}.npy'.format(iter), xreal_traj)
