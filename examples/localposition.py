
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from mpl_toolkits.mplot3d.axes3d import Axes3D

import pytransform as pytf


def main():
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    ax.set_aspect('equal')
    pytf.plot.corners(size=(4, 4, 4), ax=ax)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    origin = pytf.tf.Transform()
    pytf.plot.coordinates(origin, scale=0.5,
                          colors=pytf.plot.arm_colors('Pastel1'))

    mytf = pytf.tf.Transform(
        position=np.array([1., 1., 0.0]),
        name='parent')
    mytf.rotate(quaternion.from_rotation_vector([0, 0, -np.pi/4]))
    child_tf = pytf.tf.Transform(
        position=np.array((2.0, 1.0, 0.0)),
        name='child'
    )
    print(type(child_tf.rotation))
    child_tf.set_parent(mytf)

    pytf.plot.coordinates_all(
        mytf, ax=ax, colors=pytf.plot.arm_colors('Dark2'))

    ax.view_init(elev=45, azim=45)

    print(f'abs pos : {child_tf.position}')
    print(f'local pos : {child_tf.local_position}')

    print(f'correct: {np.array([1/np.sqrt(2),1/np.sqrt(2),0])}')

    ax.set_proj_type('ortho')
    print(type(fig))
    print(type(ax))
    plt.show()

    image_name = '3d.png'
    # plt.savefig(image_name)
    # print(image_name)


if __name__ == '__main__':
    main()
