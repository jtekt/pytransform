# https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
import numpy as np


def camera_intrinsic_matrix(fx: float, fy: float, cx: float, cy: float):
    return np.array([
        [fx, 0.0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
