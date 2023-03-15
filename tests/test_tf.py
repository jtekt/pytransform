import numpy as np
import pytest

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
