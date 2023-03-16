import matplotlib.pyplot as plt
import numpy as np
import quaternion

from .tf import Transform


def corners(center=(0, 0, 0), size=(1, 1, 1), ax: plt.Axes = None):
    if ax is None:
        ax = plt.gca()

    x, y, z = center
    w_x, w_y, w_z = size

    cube = []

    for i in [0.5, -0.5]:
        for j in [0.5, -0.5]:
            for k in [0.5, -0.5]:
                p = [x+i*w_x, y+j*w_y, z+k*w_z]
                cube.append(p)
    cube = np.array(cube)
    ax.scatter3D(cube[:, 0], cube[:, 1], cube[:, 2])


def coordinates(tf: Transform, ax: plt.Axes = None, arm_scale: float = 1.0):
    if ax is None:
        ax = plt.gca()

    # origin
    ax.scatter3D(tf.position[0], tf.position[1], tf.position[2], color='gray')

    rot_mat = quaternion.as_rotation_matrix(tf.rotation)

    unit_x = np.array([1, 0, 0]).reshape((3, 1))
    unit_y = np.array([0, 1, 0]).reshape((3, 1))
    unit_z = np.array([0, 0, 1]).reshape((3, 1))

    for u, c in zip([unit_x, unit_y, unit_z], ['#ff0000', '#00ff00', '#0000ff']):

        v = rot_mat @ (arm_scale * u)
        p = tf.position + v.ravel()
        ax.plot(
            [tf.position[0], p[0]],
            [tf.position[1], p[1]],
            [tf.position[2], p[2]],
            color=c
        )


def coordinates_all(tf: Transform, ax: plt.Axes = None, arm_scale: float = 1.0):
    if ax is None:
        ax = plt.gca()
    coordinates(tf, ax=ax, arm_scale=arm_scale)

    for child in tf.children:
        coordinates_all(child, ax=ax, arm_scale=arm_scale)
        # link
        ax.plot(
            [tf.position[0], child.position[0]],
            [tf.position[1], child.position[1]],
            [tf.position[2], child.position[2]],
            color='gray',
            linestyle='dotted'
        )