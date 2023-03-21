
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from mpl_toolkits.mplot3d.axes3d import Axes3D

import pytransform as pytf
from pytransform.joint import RevolveJoint, Limitation
from pytransform.tf import Transform
from pytransform.chain import Chain


def main():

    base_link = Transform(
        position=np.array([0.0, 0.0, 0.0]),
        name='base_link')

    arm_base = Transform(
        position=np.array((0.0, 0.0, 0.0)),
        name='arm_base'
    )

    link1 = Transform(
        position=np.array((0.0, 0.0, 2.0)),
        name='link1'
    )

    link2 = Transform(
        position=np.array((0.0, 0.0, 4.0)),
        name='link2'
    )

    link3 = Transform(
        position=np.array((0.0, 0.0, 6.0)),
        name='link3'
    )

    end_effector = Transform(
        position=np.array((0.0, 0.0, 8.0)),
        name='end_effector'
    )

    end_effector.set_parent(link3)

    j_yaw = RevolveJoint(
        parent=arm_base, child=link1,
        origin=Transform(arm_base.position, name='yaw'),
        axis=np.array([0, 0, 1]),
        limit=Limitation(2, -2)
    )

    j_pitch1 = RevolveJoint(
        parent=link1, child=link2,
        origin=Transform(link2.position, name='pitch1'),
        axis=np.array([0, 1, 0]),
        limit=Limitation(3, -3)
    )

    j_pitch2 = RevolveJoint(
        parent=link2, child=link3,
        origin=Transform(link3.position, name='pitch2'),
        axis=np.array([0, 1, 0]),
        limit=Limitation(3, -3)
    )

    robot_arm = Chain(
        links=[arm_base, link1, link2, link3, end_effector],
        joints=[j_yaw, j_pitch1, j_pitch2]
    )

    j_yaw.drive(np.pi/3)
    j_pitch1.drive(np.pi/3)
    j_pitch2.drive(np.pi/3)

    print(robot_arm.links[0].tree())

    fig = plt.figure()
    ax: plt.Axes = fig.add_subplot(projection='3d')
    ax.set_aspect('equal')
    pytf.plot.corners(size=(8, 8, 8), center=(0, 0, 4), ax=ax)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    pytf.plot.chain(robot_arm)
    ax.view_init(elev=45, azim=45)
    plt.show()


if __name__ == '__main__':
    main()
