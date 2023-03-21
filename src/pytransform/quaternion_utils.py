import numpy as np
import quaternion


def inverse(q: quaternion.quaternion):
    return np.quaternion(q.w, -q.x, -q.y, -q.z)


def rotate_toward(fm: np.ndarray, to: np.ndarray):

    ff = fm/(np.linalg.norm(fm))
    tt = to/(np.linalg.norm(to))

    # a*b*sin(theta)
    product = np.cross(ff, tt)
    angle = np.arcsin(np.linalg.norm(product))
    v = angle * (product/np.linalg.norm(product))  # rotation vector

    return quaternion.from_rotation_vector(v)
