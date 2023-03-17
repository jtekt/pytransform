
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from mpl_toolkits.mplot3d.axes3d import Axes3D

import pytransform as pytf
from pytransform.joint import RevolveJoint
from pytransform.tf import Transform


def main():
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    ax.set_aspect('equal')
    pytf.plot.corners(size=(4, 4, 4), ax=ax)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    origin = Transform(rotation=np.quaternion(1, 0, 0, 0))
    pytf.plot.coordinates(origin, scale=0.5,
                          colors=pytf.plot.arm_colors('Pastel1'))

    mytf = Transform(
        position=np.array([0.0, 1.0, 0.0]),
        name='parent')

    child_tf = Transform(
        position=np.array((0.0, 3.0, 0.0)),
        name='child'
    )

    j = RevolveJoint(
        parent=mytf, child=child_tf,
        origin=Transform(mytf.position, name='hinge'),
        axis=np.array([0, 0, 1]),
        limit=(1, -1)
    )

    print(mytf.tree())
    print(len(j.origin.children))
    pytf.plot.coordinates_all(
        mytf, ax=ax, colors=pytf.plot.arm_colors('Pastel2'))
    j.drive(np.pi/4)  # rotate around x-axis
    c = ['red', 'green', 'blue']

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
    plt.savefig(image_name)
    print(image_name)


if __name__ == '__main__':
    main()
