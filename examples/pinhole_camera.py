import matplotlib.pyplot as plt
import numpy as np

from pytransform import camera as cam
from pytransform import plot as tfplot
from pytransform.tf import Transform


def main():
    camera = Transform(name='camera')
    # GoPro
    fx = 2695.0
    fy = 2687.0
    cx = 2672.5
    cy = 1531.9

    # hfov = np.deg2rad(109.0)
    # d = cx/np.tan(0.5*hfov)
    # np.tan(0.5*hfov) = cx/fx
    hfov = 2*np.arctan2(cx, fx)
    print(f'hfov: {hfov} rad')
    cam_plane = Transform(np.array([-cx, -cy, fx]), name='cam_plane')
    cam_plane.set_parent(camera)

    world_point = Transform(np.array([cx, cy, fx*4]), name='world point')
    world_point.set_parent(camera)
    cam_mat = cam.camera_intrinsic_matrix(fx, fy, cx, cy)
    print(f'cam mat: {cam_mat}')
    # point in camera coordinate
    # pc = (cam_mat@world_point.position.reshape((3, 1))).ravel()
    pc = cam.project_to_camera_plane(world_point.position, cam_mat)

    print(f'pc: {pc}')

    cam_point = Transform(cam_plane.position, name='point on camera plane')
    cam_point.set_parent(cam_plane)
    cam_point.translate(
        np.array([pc[0], pc[1], 0]))

    # plot
    dark2 = tfplot.coordinate_cmap('Dark2')

    fig = plt.figure(figsize=(800/72, 800/72))
    ax: plt.Axes = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    tfplot.coordinates_all(
        camera, scale=1000, name_label_offset=np.array([100.0, 0.0, 0.0]))
    tfplot.rect_xy(cam_plane, size=(cx*2, cy*2),
                   center=(cx, cy), color=dark2[5])

    # ax.legend()
    # ax.view_init(elev=45, azim=45)
    ax.set_proj_type('ortho')
    ax.set_aspect('equal')
    plt.show()


if __name__ == '__main__':
    main()
