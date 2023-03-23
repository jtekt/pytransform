from __future__ import annotations

import numpy as np
from scipy import optimize

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
                 reference_tf: list[Transform],
                 target_positions: list[np.ndarray]):

        def loss_position_f(position: list):
            self.fk(position)
            loss = 0.0
            for tf, p in zip(reference_tf, target_positions):
                e = tf.position - p
                loss += np.dot(e, e)
            return loss

        initial_x = [0]*len(self.joints)
        le_lsq = optimize.minimize(
            loss_position_f,
            initial_x)
        return le_lsq
