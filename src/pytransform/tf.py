from __future__ import annotations

from typing import List, TypeVar

import numpy as np
import quaternion
from anytree import Node, RenderTree

from . import quaternion_utils as quat

# https://www.python.jp/news/wnpython311/typing2.html
Self = TypeVar("Self", bound="Transform")


class Transform():
    name: str
    __parent: Self = None  # Transform
    __children: List[Self] = []

    __position: np.ndarray
    __rotation: quaternion.quaternion
    __scale: np.ndarray

    __node: Node

    def __init__(
        self,
        position=np.array([0.0, 0.0, 0.0]),
        rotation=quaternion.one,
        scale=np.eye(3),
        name=""
    ) -> None:
        self.name = name
        self.__position = np.copy(position)
        # copy
        r = quaternion.as_float_array(rotation)
        self.__rotation = quaternion.from_float_array(r)
        self.__scale = np.copy(scale)
        self.__children = []
        self.__parent = None
        self.__node = Node(name=self.name)

    @property
    def position(self):
        return self.__position

    @property
    def local_position(self):
        if self.__parent is None:
            return self.position
        diff_rot = quat.inverse(self.__parent.rotation)*self.rotation
        diff_pos = self.position - self.__parent.position  # absolute
        p = quaternion.as_rotation_matrix(diff_rot)@diff_pos.reshape((3, 1))
        print(p.shape)
        return p.ravel()

    @property
    def rotation(self):
        return self.__rotation

    @property
    def local_rotation(self):
        if self.__parent is None:
            return self.rotation
        else:
            return quat.inverse(self.__parent.__rotation)*self.__rotation

    @property
    def scale(self):
        return self.__scale

    @property
    def children(self):
        return self.__children

    @property
    def node(self):
        return self.__node

    def set_parent(self, parent: Self):
        self.__parent = parent
        parent.__children.append(self)
        self.__node.parent = parent.node

    def translate(self, move: np.ndarray):
        self.__position += move
        for child in self.__children:
            child.translate(move)

    def rotate_around(self, rot: quaternion.quaternion, point: np.ndarray):
        self.__rotation = rot * self.__rotation

        rot_mat = quaternion.as_rotation_matrix(rot)

        radius = self.position - point

        t = rot_mat @ radius.reshape((3, 1))

        self.__position = point + t.ravel()

        for child in self.__children:
            child.rotate_around(rot, point)

    def rotate(self, rot: quaternion.quaternion):
        self.rotate_around(rot, self.position)

    def transform_direction(self, direction: np.ndarray):
        # transform a direction from local-space to world-space
        d = quaternion.as_rotation_matrix(
            self.rotation) @ direction.reshape((3, 1))
        return d.ravel()

    def tree(self):
        l = [f'{pre}{node.name}' for pre, fill, node in RenderTree(self.node)]
        return "\n\r".join(l)
