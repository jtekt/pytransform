from typing import List, TypeVar

import numpy as np
import quaternion

# https://www.python.jp/news/wnpython311/typing2.html
Self = TypeVar("Self", bound="Transform")


class Transform():
    name: str
    parent: Self = None  # Transform
    children: List[Self] = []

    __position: np.ndarray
    __rotation: quaternion.quaternion
    __scale: np.ndarray

    def __init__(
        self,
        position=np.array([0, 0, 0]),
        rotation=np.quaternion(1, 0, 0, 0),
        scale=np.eye(3),
        name=""
    ) -> None:
        self.name = name
        self.__position = position
        self.__rotation = rotation
        self.__scale = scale
        self.children = []

    @property
    def position(self):
        return self.__position

    @property
    def rotation(self):
        return self.__rotation

    @property
    def scale(self):
        return self.__scale

    def translate(self, move: np.ndarray):
        self.__position += move
        for child in self.children:
            child.translate(move)
