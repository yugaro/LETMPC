import numpy as np
from set_args import set_args
from model.vehicle import Vehicle
import matplotlib.pyplot as plt
np.random.seed(0)


def make_data(args, vehicle):
    xinits = np.array(
        [[1, 1, np.pi / 2], [1, 1, -np.pi / 2], [1, -1, np.pi / 2], [-1, 1, np.pi / 2],
         [-1, -1, np.pi / 2], [-1, 1, -np.pi /
                               2], [1, -1, -np.pi / 2], [-1, -1, -np.pi / 2],
         [0, 0, np.pi / 2], [0, 0, -np.pi / 2], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]])

    xinits = xinits * 1
    z_train = np.zeros((1, 5))
    y_train = np.zeros((1, 3))
    point_num = 5

    for i in range(xinits.shape[0]):
        x = xinits[i, :]
        for j in range(point_num):
            u = vehicle.getPIDCon(x)
            x_next = vehicle.errRK4(x, u)

            z = np.concatenate([x, u], axis=0)
            z_train = np.concatenate([z_train, z.reshape(1, -1)], axis=0)
            y_train = np.concatenate(
                [y_train, (x_next - x).reshape(1, -1)], axis=0)
            x = x_next

    fig, ax = plt.subplots()
    ax.plot(y_train[:, 0], label=r'${\rm x}$')
    ax.plot(y_train[:, 1], label=r'${\rm y}$')
    ax.plot(y_train[:, 2], label=r'$\theta$')
    ax.legend()
    plt.show()
    return z_train[1:], y_train[1:]


if __name__ == '__main__':
    args = set_args()
    vehicle = Vehicle(args)
    z_train, y_train = make_data(args, vehicle)
    np.save(args.datafile_z, z_train)
    np.save(args.datafile_y, y_train)
    print(z_train)
