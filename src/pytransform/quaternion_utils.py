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
    angle = np.arccos(np.clip(np.dot(ff, tt), -1.0, 1.0))
    if (angle - np.pi)**2 < 1e-3:
        n = np.array([1, 0, 0])
        ax = np.arccos(np.clip(np.dot(ff, np.array([1, 0, 0])), -1.0, 1.0))
        if (ax - np.pi)**2 < 1e-3:
            rx = quaternion.from_rotation_vector([np.pi*0.5, 0, 0])
            n = multiple(rx, ff)
        else:
            ry = quaternion.from_rotation_vector([0, np.pi*0.5, 0])
            n = multiple(ry, ff)
        return quaternion.from_rotation_vector(np.pi * (n))

    # a*b*sin(theta)
    product = np.cross(ff, tt)
    # angle = np.arcsin(
    #     np.clip(np.linalg.norm(product), -1.0, 1.0)
    # )

    v = angle * (product/(np.linalg.norm(product)+1e-12))  # rotation vector

    return quaternion.from_rotation_vector(v)


def from_rpy(roll: float, pitch: float, yaw: float):
    return (yaw*quaternion.z/2).exp() * \
        (pitch*quaternion.y/2).exp() * (roll*quaternion.x/2).exp()
