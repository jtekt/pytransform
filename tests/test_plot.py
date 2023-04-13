from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import quaternion

from pytransform import plot, urdf


def test_corners():
    fig = plt.figure(figsize=(800/72, 800/72))
    ax: plt.Axes = fig.add_subplot(projection='3d')
    plot.corners()
    plt.close()


def test_colors():
    plot.coordinate_cmap('Dark2')


test_urdf_items = {
    'r2d2': ('./files/r2d2.urdf', 'box'),
    'mycobot': ('./files/mycobot_style.urdf', 'hand'),
    'human': ('./files/human165.urdf', 'head_default'),
}


@pytest.mark.parametrize(
    "filename,distal_name",
    list(test_urdf_items.values()),
    ids=list(test_urdf_items.keys()))
def test_plot_urdf(filename: str, distal_name: str):
    filepath = os.path.join(
        os.path.dirname(__file__),
        filename
    )
    robot = urdf.chain_from_urdf(filepath)
    fig = plt.figure(figsize=(800/72, 800/72))
    ax: plt.Axes = fig.add_subplot(projection='3d')
    plot.chain(robot)
    plt.close()
