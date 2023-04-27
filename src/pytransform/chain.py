from __future__ import annotations

import numpy as np
import quaternion
from anytree import Node
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

        # add tag
        for l in self.links:
            l.node.link = l.name
        for j in self.joints:
            j.origin.node.joint = j.type

    @property
    def position(self):
        return [j.position for j in self.joints]

    @property
    def root(self):
        return self.links[0].root

    def get_link(self, name: str):
        candidates = [l for l in self.links if l.name == name]
        if len(candidates) == 0:
            return None
        else:
            return candidates[0]

    def get_joints(self, names: list[str] = []):

        jj = [j for j in self.joints if j.origin.name in names]
        return jj

    def fk(self, positions: list):
        # forward kinematics
        for j, p in zip(self.joints, positions):
            j.drive_to(p)

    def ik_solve(self,
                 reference_tf: list[Transform],
                 target_positions: list[np.ndarray],
                 initial_position: list[np.ndarray],
                 method: str = 'BFGS'):
        if len(target_positions) != len(reference_tf):
            raise ValueError('#target and #tf must be equal')

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

        le_lsq = optimize.minimize(
            loss_position_f,
            initial_position, method=method)
        return le_lsq

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        """bbox of links

        Returns:
            tuple[np.ndarray,np.ndarray]: min, max of bbox
        """
        positions = np.array([l.position for l in self.links])

        p_min = np.min(positions, axis=0)
        p_max = np.max(positions, axis=0)

        return p_min, p_max

    def mid_position(self):
        uppers = [j.limit.upper for j in self.joints]
        lowers = [j.limit.lower for j in self.joints]
        mid = [0.5*(u+l) for (u, l) in zip(uppers, lowers)]
        return mid

    def tree(self):
        def f(pre: str, node: Node):
            prefix = ''
            suffix = ''
            if 'joint' in node.__dict__:
                j = self.get_joints(names=[node.name])[0]
                prefix = f'[{j.type.name}] '
                if j.type != BaseJoint.Type.FIXED:
                    suffix = f' ({j.limit.lower:0.2f}--{j.limit.upper:0.2f})'

            return f'{pre}{prefix}{node.name}{suffix}'

        return self.root.tree(formatter=f)
