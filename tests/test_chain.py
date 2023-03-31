import os

import numpy as np
import pytest
import quaternion

from pytransform import urdf

test_urdf_items = {
    'r2d2': ('./files/r2d2.urdf'),
}


@pytest.mark.parametrize(
    "filename",
    list(test_urdf_items.values()),
    ids=list(test_urdf_items.keys()))
def test_read_urdf(filename: str):

    filepath = os.path.join(
        os.path.dirname(__file__),
        filename
    )
    urdf.chain_from_urdf(filepath)
