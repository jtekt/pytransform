from __future__ import annotations

import os
import time
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import quaternion
from mpl_toolkits.mplot3d.axes3d import Axes3D

import pytransform as pytf
from pytransform import urdf
from pytransform.chain import Chain
from pytransform.joint import Limitation, RevoluteJoint
from pytransform.tf import Transform


def manipulator(
    filename: str,
    target_positions: list[np.ndarray],
    is_save: bool = False,
    method="BFGS",
):
    robot = urdf.chain_from_urdf(filename)
    mi, ma = robot.bbox()
    size = ma - mi

    # visualize
    fig = plt.figure(figsize=(800 / 72, 800 / 72))
    ax: plt.Axes = fig.add_subplot(projection="3d")
    # pytf.plot.chain(robot, ax=ax, scale=size.mean()*0.1, show_label=False.as_integer_ratio,
    #                 cmap=pytf.plot.coordinate_cmap('Pastel1'))

    print(f"robot name: {robot.name}")
    print(f"#links: {len(robot.links)}")
    print(f"#joints: {len(robot.joints)}")

    hand_l = robot.get_link("left_hand_default")
    hand_r = robot.get_link("right_hand_default")
    foot_l = robot.get_link("left_foot_default")
    foot_r = robot.get_link("right_foot_default")

    for t in [hand_l, hand_r, foot_l, foot_r]:
        if t is None:
            print(f"target tf is not found")
            print(robot.tree())
            return

    bt = time.time()
    ik_result = robot.ik_solve(
        [hand_l, hand_r, foot_l, foot_r], target_positions, robot.mid_position()
    )
    ft = time.time()
    print(f"complete ik in {(ft-bt)*1000:0.3f} msec")
    print(f"ik result: \n{ik_result}")
    robot.fk(ik_result.x)
    # j_yaw.drive(angles[0])
    # j_pitch1.drive(angles[1])
    # j_pitch2.drive(angles[2])

    # print(f'manipulator structure: \n\r{robot.links[0].tree()}')

    pytf.plot.corners(size=[1.1 * size.max()] * 3, center=0.5 * (mi + ma), ax=ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    pytf.plot.chain(robot, ax=ax, scale=size.mean() * 0.1, show_label=False)
    labels = [
        "left_hand_default",
        "right_hand_default",
        "left_foot_default",
        "right_foot_default",
    ]
    for t, l in zip(target_positions, labels):
        ax.scatter(t[0], t[1], t[2], s=80, label=l)
    ax.legend()
    ax.view_init(elev=45, azim=45)
    ax.set_proj_type("ortho")
    ax.set_aspect("equal")

    if is_save:
        image_name = "ik_result_4limbs.png"
        plt.savefig(image_name)
        print(f"write {image_name}")
    else:
        plt.show()


def ik_target_template(id: int):
    template = [
        # joy
        [
            np.array([0.25 + 1, 0.1, 1.0]),
            np.array([0.25 + 1, -0.1, 0.2]),
            np.array([0.0 + 1, 0.1, -0.6]),
            np.array([0.1 + 1, -0.1, -0.4]),
        ],
        # floor
        [
            np.array([1 + 1, 0.1, 0.0]),
            np.array([1 + 1, -0.1, 0.0]),
            np.array([0.0 + 1, 0.1, -0.0]),
            np.array([0.1 + 1, -0.1, -0.0]),
        ],
    ]

    return template[id]


def main():
    default_urdf = os.path.join(
        os.path.dirname(__file__), "../tests/files/human-xyz-165.urdf"
    )

    parser = ArgumentParser()

    parser.add_argument("--urdf", default=default_urdf)

    parser.add_argument(
        "--target",
        type=int,
        help="target position template id",
        default=0,
    )
    parser.add_argument("--method", default="BFGS")
    parser.add_argument("--save", action="store_true")

    args = parser.parse_args()

    manipulator(
        args.urdf, ik_target_template(args.target), args.save, method=args.method
    )


if __name__ == "__main__":
    main()
