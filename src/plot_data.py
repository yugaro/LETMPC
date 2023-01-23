import numpy as np
from set_args import set_args
from model.vehicle import Vehicle
import matplotlib.pyplot as plt


def refTrajF(args, vehicle, xreal_traj):
    xr = np.array(args.xinit_r)
    ur = np.array([args.v_r, args.omega_r])
    ref_traj = xr
    for t in range(xreal_traj.shape[0]):
        xr_next = vehicle.realRK4(xr, ur)
        ref_traj = np.vstack([ref_traj, xr_next])
        xr = xr_next
    return ref_traj


def trackTrajF(ref_traj, xreal_traj):
    for t in range(xreal_traj.shape[0]):
        theta = ref_traj[t, 2] - xreal_traj[t, 2]
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        pos = ref_traj[t, : 2] - rotation @ xreal_traj[t, : 2]
        state = np.hstack([pos, np.array([theta])])
        if t == 0:
            track_traj = state
        else:
            track_traj = np.vstack([track_traj, state])
    return track_traj


def trackTrajPlot(ref_traj, track_traj):
    fig, ax = plt.subplots()
    ax.plot(ref_traj[:, 0], ref_traj[:, 1])
    ax.plot(track_traj[:, 0], track_traj[:, 1])
    plt.show()


if __name__ == '__main__':
    args = set_args()
    vehicle = Vehicle(args)
    # xreal_traj = np.load('../data/xreal_traj_trigger_iter0.npy')
    xreal_traj = np.load('../data/xreal_traj_periodic.npy')
    ref_traj = refTrajF(args, vehicle, xreal_traj)
    track_traj = trackTrajF(ref_traj, xreal_traj)
    trackTrajPlot(ref_traj, track_traj)
