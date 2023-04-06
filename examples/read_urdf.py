import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt

import pytransform.plot as tfplot
from pytransform import urdf


def read(filename: str, is_save: bool, scale=1.0, pose='zero'):
    robot = urdf.chain_from_urdf(filename)

    print(f'robot name: {robot.name}')
    print(f'#links: {len(robot.links)}')
    print(f'#joints: {len(robot.joints)}')
    print(robot.joints[0].origin.node)
    print(f'structure:\n{robot.tree()}')

    if pose == 'mid':
        robot.fk(robot.mid_position())
    if pose == 'max':
        robot.fk([j.limit.upper for j in robot.joints])
    if pose == 'min':
        robot.fk([j.limit.lower for j in robot.joints])
    # print(f'joints: {robot.joints}')

    fig = plt.figure(figsize=(600/72, 600/72))
    ax: plt.Axes = fig.add_subplot(projection='3d')

    mi, ma = robot.bbox()
    size = (ma-mi)
    tfplot.corners(size=1.1*size, center=0.5*(mi+ma), ax=ax)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    tfplot.chain(robot, scale=scale)
    ax.view_init(elev=45, azim=45)
    ax.set_proj_type('ortho')
    ax.set_aspect('equal')

    if is_save:
        image_name = f'{robot.name}.png'
        plt.savefig(image_name)
        print(f'write {image_name}')
    else:
        plt.show()


def main():
    parser = ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--pose',  default='zero')
    args = parser.parse_args()
    read(args.filename, args.save, args.scale, args.pose)


if __name__ == '__main__':
    main()
