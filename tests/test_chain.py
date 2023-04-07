import os

import numpy as np
import pytest
import quaternion

from pytransform import urdf

test_urdf_items = {
    'r2d2': ('./files/r2d2.urdf', 'box'),
    'mycobot': ('./files/mycobot_style.urdf', 'hand'),
    'human': ('./files/human165.urdf', 'head_default'),
}


@pytest.mark.parametrize(
    "filename,distal_name",
    list(test_urdf_items.values()),
    ids=list(test_urdf_items.keys()))
def test_read_urdf(filename: str, distal_name: str):

    filepath = os.path.join(
        os.path.dirname(__file__),
        filename
    )
    robot = urdf.chain_from_urdf(filepath)

    end_effector = robot.get_link(distal_name)
    assert end_effector is not None


test_ik_items = {
    'mycobot': ('./files/mycobot_style.urdf', 'hand'),
}


@pytest.mark.parametrize(
    "filename,distal_name",
    list(test_ik_items.values()),
    ids=list(test_ik_items.keys()))
def test_ik(filename: str, distal_name: str):

    filepath = os.path.join(
        os.path.dirname(__file__),
        filename
    )
    robot = urdf.chain_from_urdf(filepath)
    end_effector = robot.get_link(distal_name)
    assert end_effector is not None
    print(f'end effector: {end_effector.name}')
    l = robot.ik_solve(
        [end_effector],
        [np.array([0.1, 0.1, 0.1])],
        initial_position=robot.mid_position()
    )
