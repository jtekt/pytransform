import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import quaternion

from .tf import Transform


def arm_colors(palette: str):
    cmap = mpl.colormaps[palette]
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html#qualitative
    # recorder for r-g-b order
    cdict = {
        'Pastel1': (0, 2, 1),
        'Pastel2': (1, 0, 2),
        'Dark2': (1, 0, 2),
        'Set1': (0, 2, 1),
        'Set2': (1, 0, 2)
    }

    if palette in cdict:
        i = cdict[palette]
        return (cmap(i[0]), cmap(i[1]), cmap(i[2]))

    return (cmap(0), cmap(1), cmap(2))


def corners(center=(0, 0, 0), size=(1, 1, 1), ax: plt.Axes = None, color=(0.1, 0.1, 0.1, 0.1)):
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
    ax.scatter3D(cube[:, 0], cube[:, 1], cube[:, 2], color=color)


def coordinates(
        tf: Transform,
        ax: plt.Axes = None,
        scale: float = 1.0,
        colors: list = arm_colors('Set2'),
        show_name: bool = True,
        name_label_offset=np.array([0, 0, 1])):
    if ax is None:
        ax = plt.gca()

    # origin
    ax.scatter3D(tf.position[0], tf.position[1], tf.position[2], color='gray')

    rot_mat = quaternion.as_rotation_matrix(tf.rotation)

    unit_x = np.array([1, 0, 0]).reshape((3, 1))
    unit_y = np.array([0, 1, 0]).reshape((3, 1))
    unit_z = np.array([0, 0, 1]).reshape((3, 1))

    units = [unit_x, unit_y, unit_z]
    for u, c in zip(units, colors):

        v = rot_mat @ (scale * u)
        p = tf.position + v.ravel()
        ax.plot(
            [tf.position[0], p[0]],
            [tf.position[1], p[1]],
            [tf.position[2], p[2]],
            color=c
        )
    if show_name and (len(tf.name) > 0):
        label_pos = tf.position+name_label_offset
        ax.text(label_pos[0], label_pos[1], label_pos[2],
                tf.name)


def coordinates_all(
        tf: Transform, ax: plt.Axes = None, scale: float = 1.0,
        colors: list = arm_colors('Set2')):
    if ax is None:
        ax = plt.gca()
    coordinates(tf, ax, scale, colors)

    for child in tf.children:
        coordinates_all(child, ax, scale, colors)
        # link
        ax.plot(
            [tf.position[0], child.position[0]],
            [tf.position[1], child.position[1]],
            [tf.position[2], child.position[2]],
            color='gray',
            linestyle='dotted'
        )
