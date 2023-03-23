import numpy as np
import pytest
import quaternion

from pytransform import quaternion_utils as quat
from pytransform.tf import Transform


def test_transform():
    my_tf = Transform()

    # property accesses
    my_tf.position
    my_tf.rotation
    my_tf.scale
    my_tf.local_position
    my_tf.local_rotation


def test_id():
    p = np.array([0, 1, 1])
    r = quaternion.from_vector_part([1, 0, 0])

    print(f'src: {id(p)}, {id(r)}')

    t = Transform(p, r)

    print(f'tf: {id(t.position)}, {id(t.rotation)}')

    assert id(p) != id(t.position)
    assert id(r) != id(t.rotation)


test_translate_items = {
    'base': (np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([2, 0, 0])),
}


@pytest.mark.parametrize(
    "parent_pos,child_pos, move",
    list(test_translate_items.values()),
    ids=list(test_translate_items.keys()))
def test_translate(parent_pos, child_pos, move):
    parent_tf = Transform(name="root", position=parent_pos)
    child_tf = Transform(
        position=child_pos, name="link")
    child_tf.set_parent(parent_tf)
    parent_tf.translate(move)
    diff = parent_tf.position - (parent_pos+move)
    assert np.linalg.norm(diff) < 1e-3
    diff = parent_tf.children[0].position - (child_pos+move)
    assert np.linalg.norm(diff) < 1e-3


@pytest.mark.parametrize(
    "parent_pos,child_pos, target",
    list(test_translate_items.values()),
    ids=list(test_translate_items.keys()))
def test_translate_to(parent_pos, child_pos, target):
    parent_tf = Transform(name="root", position=parent_pos)
    child_tf = Transform(
        position=child_pos, name="link")
    child_tf.set_parent(parent_tf)
    parent_tf.translate_to(target)
    diff = parent_tf.position - target
    assert np.linalg.norm(diff) < 1e-3


test_rotate_items = {
    'base': (
        np.quaternion(1, 0, 0, 0),  # parent
        np.quaternion(1, 0, 0, 0),  # child rot
        np.array([0., 0., 1.]),  # child pos
        quaternion.from_rotation_vector(np.array((np.pi/2, 0, 0))),  # rot
        np.array([0., -1., 0.])  # child answer
    ),
}


@pytest.mark.parametrize(
    "parent_rot, child_rot, child_pos, rot, child_ans",
    list(test_rotate_items.values()),
    ids=list(test_rotate_items.keys()))
def test_rotate(
        parent_rot, child_rot, child_pos, rot, child_ans):
    parent_tf = Transform(
        name="root", rotation=parent_rot)
    child_tf = Transform(
        position=child_pos,
        rotation=child_rot, name="link")
    child_tf.set_parent(parent_tf)

    parent_tf.rotate(rot)

    assert parent_tf.rotation == rot*parent_rot
    assert parent_tf.children[0].rotation == rot*child_rot

    diff = child_tf.position - child_ans
    print(child_tf.position)
    assert np.linalg.norm(diff) < 1e-3

    # property accesses
    child_tf.position
    child_tf.rotation
    child_tf.scale
    child_tf.local_position
    child_tf.local_rotation


@pytest.mark.parametrize(
    "parent_rot, child_rot, child_pos, rot, child_ans",
    list(test_rotate_items.values()),
    ids=list(test_rotate_items.keys()))
def test_rotate_to(
        parent_rot, child_rot, child_pos, rot, child_ans):
    parent_tf = Transform(
        name="root", rotation=parent_rot)
    child_tf = Transform(
        position=child_pos,
        rotation=child_rot, name="link")
    child_tf.set_parent(parent_tf)

    parent_tf.rotate_to(rot)

    diff = quaternion.as_float_array(
        parent_tf.rotation)-quaternion.as_float_array(rot)
    assert np.linalg.norm(diff) < 1e-3


def test_localpos():
    mytf = Transform(position=np.array([1., 1., 0.0]))
    mytf.rotate(quaternion.from_rotation_vector([0, 0, -np.pi/4]))
    child_tf = Transform(
        position=np.array((2.0, 1.0, 0.0))
    )
    child_tf.set_parent(mytf)

    ans = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])

    error = ans - child_tf.local_position
    assert np.linalg.norm(error) < 1e-6

    d = mytf.transform_direction(child_tf.local_position)
    print(f'd: {d}')
    err2 = d - (child_tf.position - mytf.position)
    assert np.linalg.norm(err2) < 1e-6
