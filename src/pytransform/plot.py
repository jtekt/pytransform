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
        "Pastel1": (0, 2, 1),
        "Pastel2": (1, 0, 2),
        "Dark2": (1, 0, 2),
        "Set1": (0, 2, 1),
        "Set2": (1, 0, 2),
    }
    n = 8
    idx = [i for i in range(n)]

    if palette in cdict:
        idx[0] = cdict[palette][0]
        idx[1] = cdict[palette][1]
        idx[2] = cdict[palette][2]

    return [cmap(i) for i in idx]


def corners(
    center=(0, 0, 0), size=(1, 1, 1), ax: plt.Axes = None, color=(0.1, 0.1, 0.1, 0.1)
):
    if ax is None:
        ax = plt.gca()

    x, y, z = center
    w_x, w_y, w_z = size

    cube = []

    for i in [0.5, -0.5]:
        for j in [0.5, -0.5]:
            for k in [0.5, -0.5]:
                p = [x + i * w_x, y + j * w_y, z + k * w_z]
                cube.append(p)
    cube = np.array(cube)
    ax.scatter3D(cube[:, 0], cube[:, 1], cube[:, 2], color=color)


def rect_xy(
    tf: Transform,
    size: tuple[float, float],
    center: tuple[float, float],
    ax: plt.Axes = None,
    color=(0.1, 0.1, 0.1, 0.1),
):
    if ax is None:
        ax = plt.gca()
    cc = np.array([center[0], center[1], 0])
    cw = tf.position + cc
    ex = tf.transform_direction(np.array([1.0, 0, 0]))
    ey = tf.transform_direction(np.array([0, 1.0, 0]))
    rect_corners = []
    w, h = size

    for k in ([-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5]):
        # p = np.array([i*w*ex, j*h*ey, 0])
        rect_corners.append(cw + k[0] * w * ex + k[1] * h * ey)
    rect_corners = np.array(rect_corners)
    ax.plot(rect_corners[:, 0], rect_corners[:, 1], rect_corners[:, 2], color=color)
    ax.plot(
        [rect_corners[-1, 0], rect_corners[0, 0]],
        [rect_corners[-1, 1], rect_corners[0, 1]],
        [rect_corners[-1, 2], rect_corners[0, 2]],
        color=color,
    )
    # cross
    for i in range(2):
        ax.plot(
            [rect_corners[i, 0], rect_corners[i + 2, 0]],
            [rect_corners[i, 1], rect_corners[i + 2, 1]],
            [rect_corners[i, 2], rect_corners[i + 2, 2]],
            color=color,
        )


def coordinates(
    tf: Transform,
    ax: plt.Axes = None,
    scale: float = 1.0,
    colors: list = coordinate_cmap("Set2"),
    show_name: bool = True,
    linewidth: float = 1.0,
    name_label_offset=np.array([0, 0, 1]),
):
    if ax is None:
        ax = plt.gca()

    # origin
    ax.scatter3D(tf.position[0], tf.position[1], tf.position[2], color="gray")

    rot_mat = quaternion.as_rotation_matrix(tf.rotation)

    unit_x = np.array([scale, 0, 0]).reshape((3, 1))
    unit_y = np.array([0, scale, 0]).reshape((3, 1))
    unit_z = np.array([0, 0, scale]).reshape((3, 1))

    units = [unit_x, unit_y, unit_z]
    for u, c in zip(units, colors):
        v = rot_mat @ u
        p = tf.position + v.ravel()
        ax.plot(
            [tf.position[0], p[0]],
            [tf.position[1], p[1]],
            [tf.position[2], p[2]],
            color=c,
            linewidth=linewidth,
        )
    if show_name and (len(tf.name) > 0):
        label_pos = tf.position + name_label_offset
        ax.text(label_pos[0], label_pos[1], label_pos[2], tf.name)


def coordinates_all(
    tf: Transform,
    ax: plt.Axes = None,
    scale: float = 1.0,
    colors: list = coordinate_cmap("Set2"),
    name_label_offset=np.array([0, 0, 1.0]),
):
    if ax is None:
        ax = plt.gca()
    coordinates(tf, ax, scale, colors, name_label_offset=name_label_offset)

    for child in tf.children:
        coordinates_all(child, ax, scale, colors, name_label_offset=name_label_offset)
        # link
        ax.plot(
            [tf.position[0], child.position[0]],
            [tf.position[1], child.position[1]],
            [tf.position[2], child.position[2]],
            color="gray",
            linestyle="dotted",
        )


def joint(j: jnt.BaseJoint, ax: plt.Axes = None, color=(1, 0, 0), scale: float = 1.0):
    if j.type == jnt.BaseJoint.Type.FIXED:
        fixed_joint(j, ax, color, scale)
    elif j.type == jnt.BaseJoint.Type.REVOLUTE:
        rev_joint(j, ax, color, scale=scale)
    elif j.type == jnt.BaseJoint.Type.CONTINUOUS:
        rev_joint(j, ax, color, scale=scale, num_rings=2)
    elif j.type == jnt.BaseJoint.Type.PRISMATIC:
        prismatic_joint(j, ax, color, scale)
    else:
        print(f"{j.type} cannot be visualized")


def fixed_joint(
    j: jnt.BaseJoint, ax: plt.Axes = None, color=(0.5, 0, 0), scale: float = 1.0
):
    if ax is None:
        ax = plt.gca()
    ax.scatter(
        j.origin.position[0],
        j.origin.position[1],
        j.origin.position[2],
        color=color,
        marker="s",
    )


def rev_joint(
    j: jnt.BaseJoint,
    ax: plt.Axes = None,
    color=(0.5, 0, 0),
    scale: float = 1.0,
    resolution: int = 16,
    num_rings: int = 1,
):
    if ax is None:
        ax = plt.gca()

    # origin
    m = "o"
    if num_rings >= 2:
        m = "*"
    ax.scatter(
        j.origin.position[0],
        j.origin.position[1],
        j.origin.position[2],
        color=color,
        marker=m,
    )

    # rotation axis
    # rotation axis in world space
    rot_axis = scale * j.origin.transform_direction(j.axis)
    start = j.origin.position - rot_axis
    end = j.origin.position + rot_axis
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color=color
        # linestyle='dotted'
    )

    # ax.text(end[0], end[1], end[2], j.name, zdir=rot_axis)

    # ring
    for s in range(num_rings):
        plot_ring(
            (s + 1 / num_rings) * scale,
            j.origin.position,
            rot_axis,
            ax,
            color,
            linestyle="dotted",
        )

    # b = np.array(j.axis[:-1])


def prismatic_joint(
    j: jnt.BaseJoint, ax: plt.Axes = None, color=(0.5, 0, 0), scale: float = 1.0
):
    if ax is None:
        ax = plt.gca()

    # origin
    ax.scatter(
        j.origin.position[0],
        j.origin.position[1],
        j.origin.position[2],
        color=color,
        marker="d",
    )

    # axis
    slide_axis = scale * j.origin.transform_direction(j.axis)
    start = j.origin.position - slide_axis
    end = j.origin.position + slide_axis
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        [start[2], end[2]],
        color=color
        # linestyle='dotted'
    )
    # https://matplotlib.org/stable/gallery/mplot3d/text3d.html
    # slide guide
    plot_ring(scale, start, slide_axis, ax, color, resolution=4, linestyle="dotted")
    plot_ring(scale, end, slide_axis, ax, color, resolution=4, linestyle="dotted")


def chain(
    ch: Chain,
    ax: plt.Axes = None,
    cmap=coordinate_cmap("Dark2"),
    scale: float = 1.0,
    show_label=True,
):
    if ax is None:
        ax = plt.gca()

    for link in ch.links:
        coordinates(
            link,
            ax=ax,
            colors=cmap,
            scale=scale,
            show_name=show_label,
            name_label_offset=np.array([0, 0, scale]),
        )
        for child in link.children:
            # link
            ax.plot(
                [link.position[0], child.position[0]],
                [link.position[1], child.position[1]],
                [link.position[2], child.position[2]],
                color="gray",
                linestyle="dotted",
            )
    for j in ch.joints:
        joint(j, ax, color=cmap[3], scale=scale)


def ring_point(r: float = 1, resolution: int = 16):
    p = [
        r * np.array([np.cos(angle), np.sin(angle), 0])
        for angle in np.linspace(0, 2 * np.pi, num=resolution, endpoint=False)
    ]

    return np.array(p)


def plot_ring(
    radius: float = 1.0,
    center=np.array([0, 0, 0]),
    axis=np.array([0, 0, 1]),
    ax: plt.Axes = None,
    color=(0.5, 0, 0),
    resolution: int = 16,
    linestyle="dotted",
):
    if ax is None:
        ax = plt.gca()

    q = quat.rotate_toward(np.array([0, 0, 1]), axis)

    pp = ring_point(radius, resolution=resolution)

    pr = [quat.multiple(q, p) for p in pp]
    pr.append(pr[0])
    pr = np.array(pr) + center

    ax.plot(pr[:, 0], pr[:, 1], pr[:, 2], color=color, linestyle=linestyle)
