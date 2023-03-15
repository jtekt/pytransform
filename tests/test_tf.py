from pytransform.tf import Transform


def test_transform():
    my_tf = Transform()

    # property accesses
    my_tf.position
    my_tf.rotation
    my_tf.scale
