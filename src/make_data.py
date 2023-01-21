import numpy as np
from set_args import set_args
from model.vehicle import Vehicle
import matplotlib.pyplot as plt
np.random.seed(0)


def make_data(args, vehicle):
    xinits = np.array(
        [[-3, -3, np.pi], [-1, -2, -np.pi / 2], [-2, -1, -np.pi], [-1, -1, np.pi / 2]])

    xinits = xinits
    z_train = np.zeros((1, 5))
    y_train = np.zeros((1, 3))
    point_num = 8

    for i in range(xinits.shape[0]):
        x = xinits[i, :]
        for j in range(point_num):
            u = vehicle.getPIDConRand(x)
            x_next = vehicle.errRK4(x, u)

            z = np.hstack([x, u])
            z_train = np.vstack([z_train, z])
            y_train = np.vstack([y_train, (x_next - x)])
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
    print(z_train)
    print(y_train)
    np.save(args.datafile_z, z_train)
    np.save(args.datafile_y, y_train)
