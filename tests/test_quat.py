import numpy as np
import pytest
import quaternion

from pytransform import quaternion_utils as quat


def test_inverse():
    q1 = quaternion.from_rotation_vector([2.0, 6.0, 5.0])
    iq = quat.inverse(q1)
    i = iq*q1
    a = quaternion.as_float_array(i)
    error = a-np.array([1.0, 0.0, 0.0, 0.0])
    assert np.linalg.norm(error) < 1e-6


direction_items = {
    'a': (np.array([1, 0, 0]), np.array([0, 1, 0])),
    'b': (np.array([0, 1, 0]), np.array([0, 0, 1])),
    'c': (np.array([1, 1, 1]), np.array([1, -1, 1])),
    'd': (np.array([1, 0, 0]), np.array([1, 1, 0])),
    'e': (np.array([1, 2, 3]), np.array([4, 5, 6])),
}


@pytest.mark.parametrize(
    "a,b",
    list(direction_items.values()),
    ids=list(direction_items.keys()))
def test_toward(a, b):
    q = quat.rotate_toward(a, b)

    an = a/(np.linalg.norm(a))
    bn = b/(np.linalg.norm(b))
    theta = np.arccos(np.dot(an, bn))
    v = quaternion.as_rotation_vector(q)
    print(f'rotation vector: {v}')
    print(f'theta: {theta} rad')
    assert (theta - np.linalg.norm(v))**2 < 1e-3
    assert np.dot(a, v) < 1e-3
    assert np.dot(b, v) < 1e-3

    # bb = quaternion.as_rotation_matrix(q) @ a.reshape((3, 1))
    bb = quat.multiple(q, a)

    error = np.linalg.norm(np.cross(b, bb))

    print(f'b={b}')
    print(f'bb={bb}, {bb.shape}')
    assert error < 1e-3
