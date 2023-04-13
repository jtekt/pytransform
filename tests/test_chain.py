from __future__ import annotations

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
    'mycobot': ('./files/mycobot_style.urdf', ['hand'], [np.array([0.1, 0.1, 0.1])]),
    'human': ('./files/human-xyz-165.urdf',
              ['left_hand_default', 'right_hand_default',
                  'left_foot_default', 'right_foot_default'],
              [
                  np.array([0.25+1, 0.1, 1.0]),
                  np.array([0.25+1, -0.1, 0.2]),
                  np.array([0.0+1, 0.1, -0.6]),
                  np.array([0.1+1, -0.1, -0.4])
              ]),
}


@pytest.mark.parametrize(
    "filename,distal_names,targets",
    list(test_ik_items.values()),
    ids=list(test_ik_items.keys()))
def test_ik(filename: str, distal_names: list(str), targets: list(np.ndarray)):

    filepath = os.path.join(
        os.path.dirname(__file__),
        filename
    )
    robot = urdf.chain_from_urdf(filepath)

    ref_tfs = [robot.get_link(n)
               for n in distal_names if robot.get_link(n) is not None]

    assert len(ref_tfs) >= 0
    l = robot.ik_solve(
        ref_tfs,
        targets,
        initial_position=robot.mid_position()
    )
