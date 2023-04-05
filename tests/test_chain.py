import os

import numpy as np
import pytest
import quaternion

from pytransform import urdf

test_urdf_items = {
    'r2d2': ('./files/r2d2.urdf', 'box'),
    'mycobot': ('./files/mycobot_style.urdf', 'hand'),
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
