import numpy as np
import quaternion


def identity():
    return np.quaternion(1, 0, 0, 0)


def inverse(q: quaternion.quaternion):
    return np.quaternion(q.w, -q.x, -q.y, -q.z)


def multiple(q: quaternion.quaternion, v: np.ndarray):
    mat = quaternion.as_rotation_matrix(q)
    u = mat @ v.reshape((3, 1))
    return u.ravel()


def rotate_toward(fm: np.ndarray, to: np.ndarray):

    ff = fm/(np.linalg.norm(fm))
    tt = to/(np.linalg.norm(to))

    # a*b*sin(theta)
    product = np.cross(ff, tt)

    angle = np.arcsin(
        np.clip(np.linalg.norm(product), -1.0, 1.0)
    )
    v = angle * (product/(np.linalg.norm(product)+1e-12))  # rotation vector

    return quaternion.from_rotation_vector(v)


def from_rpy(roll: float, pitch: float, yaw: float):
    return (yaw*quaternion.z/2).exp() * \
        (pitch*quaternion.y/2).exp() * (roll*quaternion.x/2).exp()
