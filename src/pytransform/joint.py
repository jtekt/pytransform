# http://wiki.ros.org/urdf/XML/joint

from enum import Enum, auto
from typing import NamedTuple

import numpy as np
import quaternion

from .tf import Transform


class Limitation(NamedTuple):
    upper: float
    lower: float


class BaseJoint():
    class Type(Enum):
        MISC = auto()
        REVOLVE = auto()
        PRISMATIC = auto()

    __position: np.ndarray
    __type: Type

    def __init__(self,
                 parent: Transform,
                 child: Transform,
                 origin: Transform,
                 axis: np.ndarray = np.array([1, 0, 0]),
                 limit: Limitation = Limitation(0.0, 0.0)
                 ) -> None:
        self.parent = parent
        self.child = child

        self.origin = origin
        self.origin.set_parent(self.parent)
        self.child.set_parent(self.origin)

        self.axis = axis/np.linalg.norm(axis)
        self.limit = limit

        self.__position = np.zeros(0)
        # self.__type = self.Type.MISC

    @property
    def position(self):
        return self.__position

    def set_position(self, new: np.ndarray):
        self.__position = new

    @property
    def type(self):
        return self.__type

    def set_type(self, t: Type):
        self.__type = t

    def drive(self):
        # please overwrite
        assert True


class RevolveJoint(BaseJoint):
    def __init__(self, parent: Transform, child: Transform, origin: Transform, axis: np.ndarray = np.array([1, 0, 0]), limit: Limitation = Limitation(0, 0)) -> None:
        super().__init__(parent, child, origin, axis, limit)
        self.set_position(np.zeros(0))
        self.set_type(self.Type.REVOLVE)

        assert self.type

    def drive(self, angle: float):

        p = self.position + angle

        if p > self.limit.upper:
            return
        if p < self.limit.lower:
            return

        v = angle*self.axis
        a = self.origin.transform_direction(v)
        # a: rotation vector in world-space
        q = quaternion.from_rotation_vector(a)
        self.child.rotate_around(q, self.origin.position)
        self.set_position(p)


class PrismaticJoint(BaseJoint):
    def __init__(self, parent: Transform, child: Transform, origin: Transform, axis: np.ndarray = np.array([1, 0, 0]), limit: Limitation = Limitation(0, 0)) -> None:
        super().__init__(parent, child, origin, axis, limit)

    def drive(self, move: float):

        p = self.position + move
        if p > self.limit.upper:
            return
        if p < self.limit.lower:
            return
        t = move * self.origin.transform_direction(self.axis)
        self.child.translate(t)
        self.set_position(p)