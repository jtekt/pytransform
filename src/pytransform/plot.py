import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import quaternion

from . import joint as jnt
from . import quaternion_utils as quat
from .chain import Chain
from .tf import Transform


def coordinate_cmap(palette: str):
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
    n = 8
    idx = [i for i in range(n)]

    if palette in cdict:
        idx[0] = cdict[palette][0]
        idx[1] = cdict[palette][1]
        idx[2] = cdict[palette][2]

    return [cmap(i) for i in idx]


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
        colors: list = coordinate_cmap('Set2'),
        show_name: bool = True,
        linewidth: float = 1.0,
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
            color=c,
            linewidth=linewidth
        )
    if show_name and (len(tf.name) > 0):
        label_pos = tf.position+name_label_offset
        ax.text(label_pos[0], label_pos[1], label_pos[2],
                tf.name)


def coordinates_all(
        tf: Transform, ax: plt.Axes = None, scale: float = 1.0,
        colors: list = coordinate_cmap('Set2')):
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


def joint(j: jnt.BaseJoint, ax: plt.Axes = None, color=(1, 0, 0)):
    if j.type == jnt.BaseJoint.Type.REVOLVE:
        rev_joint(j, ax, color)
    else:
        print(f'{j.type} cannot be visualized')


def rev_joint(j: jnt.BaseJoint,
              ax: plt.Axes = None,
              color=(.5, 0, 0),
              resolution: int = 16):
    if ax is None:
        ax = plt.gca()

    # origin
    ax.scatter(j.origin.position[0], j.origin.position[1],
               j.origin.position[2], color=color)

    # rotation axis
    # rotation axis in world space
    rot_axis = j.origin.transform_direction(j.axis)
    start = j.origin.position - rot_axis
    end = j.origin.position + rot_axis
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color=color
        # linestyle='dotted'
    )

    # ring
    q = quat.rotate_toward(np.array([0, 0, 1]), rot_axis)
    pp = ring_point(resolution=resolution)

    pr = [(quaternion.as_rotation_matrix(q)@p).ravel() for p in pp]
    pr = np.array(pr)+j.origin.position

    ax.plot(
        pr[:, 0],
        pr[:, 1],
        pr[:, 2],
        color=color
        # linestyle='dotted'
    )

    # b = np.array(j.axis[:-1])


def chain(ch: Chain, ax: plt.Axes = None,
          cmap=coordinate_cmap('Dark2')):
    if ax is None:
        ax = plt.gca()

    for link in ch.links:
        coordinates(link, ax=ax, colors=cmap)
        for child in link.children:
            # link
            ax.plot(
                [link.position[0], child.position[0]],
                [link.position[1], child.position[1]],
                [link.position[2], child.position[2]],
                color='gray',
                linestyle='dotted'
            )
    for j in ch.joints:
        joint(j, ax, color=cmap[3])


def ring_point(r: float = 1, resolution: int = 16):
    p = [r*np.array([np.cos(angle), np.sin(angle), 0])
         for angle in np.linspace(0, 2*np.pi, num=resolution, endpoint=False)]

    return np.array(p)
