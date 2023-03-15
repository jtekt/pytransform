import numpy as np

from pytransform.tf import Transform


def test_transform():
    my_tf = Transform()

    # property accesses
    my_tf.position
    my_tf.rotation
    my_tf.scale


def test_translate():
    parent_tf = Transform(name="root")
    child_tf = Transform(position=np.array([1, 0, 0]), name="link")

    parent_tf.children.append(child_tf)
    parent_tf.translate(np.array([2, 0, 0]))

    assert np.array_equal(parent_tf.position, np.array([2, 0, 0]))
    assert np.array_equal(parent_tf.children[0].position, np.array([3, 0, 0]))
