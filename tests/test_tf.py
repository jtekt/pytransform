import numpy as np
import pytest
import quaternion

from pytransform.tf import Transform


def test_transform():
    my_tf = Transform()

    # property accesses
    my_tf.position
    my_tf.rotation
    my_tf.scale


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

    parent_tf.children.append(child_tf)
    parent_tf.translate(move)

    assert np.array_equal(parent_tf.position, parent_pos+move)
    assert np.array_equal(parent_tf.children[0].position, child_pos+move)


test_rotate_items = {
    'base': (
        np.quaternion(1, 0, 0, 0),  # parent
        quaternion.from_euler_angles(1, 0, 0),  # child
        quaternion.from_euler_angles(1, 0, 0)),  # rot
}


@pytest.mark.parametrize(
    "parent_rot,child_rot, rot",
    list(test_rotate_items.values()),
    ids=list(test_rotate_items.keys()))
def test_rotate(
        parent_rot, child_rot, rot):
    parent_tf = Transform(
        name="root", rotation=parent_rot)
    child_tf = Transform(
        rotation=child_rot, name="link")
    parent_tf.children.append(child_tf)

    parent_tf.rotate(rot)

    assert parent_tf.rotation == rot*parent_rot
    assert parent_tf.children[0].rotation == rot*child_rot
