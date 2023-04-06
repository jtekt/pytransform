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


def manipulator(filename: str,
                target_positions: list[np.ndarray],
                is_save: bool = False):

    robot = urdf.chain_from_urdf(filename)
    print(f'robot name: {robot.name}')
    print(f'#links: {len(robot.links)}')
    print(f'#joints: {len(robot.joints)}')

    hand_l = robot.get_link('l_wrist_default')
    hand_r = robot.get_link('r_wrist_default')

    bt = time.time()
    ik_result = robot.ik_solve(
        [hand_l, hand_r],
        target_positions,
        robot.mid_position()
    )
    ft = time.time()
    print(f'complete ik in {(ft-bt)*1000:0.3f} msec')
    print(f'ik result: \n{ik_result}')
    robot.fk(ik_result.x)
    # j_yaw.drive(angles[0])
    # j_pitch1.drive(angles[1])
    # j_pitch2.drive(angles[2])

    # print(f'manipulator structure: \n\r{robot.links[0].tree()}')

    # visualize
    fig = plt.figure(figsize=(800/72, 800/72))
    ax: plt.Axes = fig.add_subplot(projection='3d')

    mi, ma = robot.bbox()
    size = (ma-mi)
    pytf.plot.corners(size=1.1*size, center=0.5*(mi+ma), ax=ax)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    pytf.plot.chain(robot, ax=ax, scale=size.mean()*0.1)
    for i, t in enumerate(target_positions):
        ax.scatter(
            t[0], t[1], t[2],
            s=80, label=f'target {i:02d}')
    ax.legend()
    ax.view_init(elev=45, azim=45)
    ax.set_proj_type('ortho')
    ax.set_aspect('equal')

    if is_save:
        image_name = 'ik_result.png'
        plt.savefig(image_name)
        print(f'write {image_name}')
    else:
        plt.show()


def main():

    default_urdf = os.path.join(
        os.path.dirname(__file__),
        '../tests/files/human165.urdf'
    )

    parser = ArgumentParser()

    parser.add_argument('--urdf', default=default_urdf)

    parser.add_argument('--target', type=float, nargs=3,
                        help='target position for end effector',
                        default=[1.5, 1.5, 4.0])
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()

    targets = [
        np.array([0.25, 0.1, 0.8]),
        np.array([0.25, -0.1, 0.8])
    ]

    manipulator(args.urdf, targets, args.save)


if __name__ == '__main__':
    main()
