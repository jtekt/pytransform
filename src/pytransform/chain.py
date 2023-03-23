from __future__ import annotations

import numpy as np

from .joint import BaseJoint
from .tf import Transform


class Chain():
    links: list[Transform] = []
    joints: list[BaseJoint] = []

    def __init__(self,
                 links: list[Transform],
                 joints: list[BaseJoint]) -> None:
        self.links = links
        self.joints = joints

    @property
    def position(self):
        return [j.position for j in self.joints]

    def fk(self, positions: list):
        for j, p in zip(self.joints, positions):
            j.drive_to(p)

    def ik_solve(self,
                 end_effector_tf: Transform,
                 target_position: np.ndarray):

        def loss_position_f(x): return x

        pass
