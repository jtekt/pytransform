from __future__ import annotations

import numpy as np
import quaternion
from scipy import optimize

from . import quaternion_utils as quat
from .joint import BaseJoint
from .tf import Transform


class Chain():
    name: str = ''
    links: list[Transform] = []
    joints: list[BaseJoint] = []

    def __init__(self,
                 links: list[Transform],
                 joints: list[BaseJoint],
                 name: str = '') -> None:
        self.links = links
        self.joints = joints
        self.name = name

    @property
    def position(self):
        return [j.position for j in self.joints]

    @property
    def root(self):
        return self.links[0].root

    def fk(self, positions: list):
        # forward kinematics
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

            # for tf, r in zip(reference_tf,target_rotations):
            #     q_e = quat.inverse(r)* tf.rotation
            #     v = quaternion.as_rotation_vector(q_e)
            #     loss += np.dot(v,v)

            return loss

        initial_x = [0]*len(self.joints)
        le_lsq = optimize.minimize(
            loss_position_f,
            initial_x)
        return le_lsq
