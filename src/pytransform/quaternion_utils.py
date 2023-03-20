import numpy as np
import quaternion


def inverse(q: quaternion.quaternion):
    return np.quaternion(q.w, -q.x, -q.y, -q.z)


def rotate_toward(fm: np.ndarray, to: np.ndarray):

    product = np.cross(fm, to)
    # a*b*sin(theta)
    p = np.linalg.norm(product)
    s = (p)/(np.linalg.norm(fm)*np.linalg.norm(to)+1e-9)
    angle = np.arcsin(s)
    v = angle * (product/p)  # rotation vector

    return quaternion.from_rotation_vector(v)
