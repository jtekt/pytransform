from typing import List, TypeVar

import numpy as np
import quaternion

# https://www.python.jp/news/wnpython311/typing2.html
Self = TypeVar("Self", bound="Transform")


def inverse_q(q: quaternion.quaternion):
    return np.quaternion(q.w, -q.x, -q.y, -q.z)


class Transform():
    name: str
    __parent: Self = None  # Transform
    children: List[Self] = []

    __position: np.ndarray
    __rotation: quaternion.quaternion
    __scale: np.ndarray

    def __init__(
        self,
        position=np.array([0.0, 0.0, 0.0]),
        rotation=np.quaternion(1.0, 0., 0., 0.),
        scale=np.eye(3),
        name=""
    ) -> None:
        self.name = name
        self.__position = np.copy(position)
        self.__rotation = np.copy(rotation)
        self.__scale = np.copy(scale)
        self.children = []
        self.__parent = None

    @property
    def position(self):
        return self.__position

    @property
    def local_position(self):
        if self.parent is None:
            return self.__position
        else:
            return self.__position - self.parent.__position

    @property
    def rotation(self):
        return self.__rotation

    @property
    def local_rotation(self):
        if self.parent is None:
            return self.rotation
        else:
            return inverse_q(self.parent.__rotation)*self.__rotation

    @property
    def scale(self):
        return self.__scale

    def translate(self, move: np.ndarray):
        self.__position += move
        for child in self.children:
            child.translate(move)

    def rotate_around(self, rot: quaternion.quaternion, point: np.ndarray):
        self.__rotation = rot * self.__rotation

        rot_mat = quaternion.as_rotation_matrix(rot)

        diff = self.position - point

        t = rot_mat @ diff.reshape((3, 1))

        self.__position += t.ravel()

        for child in self.children:
            child.rotate_around(rot, point)

    def rotate(self, rot: quaternion.quaternion):
        self.rotate_around(rot, np.zeros(3))
