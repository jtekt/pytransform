
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from mpl_toolkits.mplot3d import Axes3D

import pytransform as pytf


def main():

    fig = plt.figure(figsize=(512/72, 512/72))
    ax: plt.Axes = fig.add_subplot(projection='3d')
    ax.set_aspect('equal')

    s = 1.0
    pytf.plot.corners(size=(s*1.1, s*1.1, s*1.1), ax=ax)

    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')

    origin = pytf.tf.Transform()

    c = pytf.plot.coordinate_cmap('Dark2')
    pytf.plot.coordinates(origin, scale=s,
                          colors=c, linewidth=4.0)
    ax.tick_params(labelbottom=False, labelleft=False,
                   labelright=False, labeltop=False)
    # ax.set_axis_off()

    el = np.arctan2(1, np.sqrt(2))
    ax.view_init(elev=np.rad2deg(el), azim=45)
    ax.set_proj_type('ortho')
    plt.show()

    image_name = 'logo.png'
    # plt.show()
    plt.savefig(image_name)
    print(f'write {image_name}')


if __name__ == '__main__':
    main()
