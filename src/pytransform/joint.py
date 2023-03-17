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
        REVOLVE = auto()

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

    def drive(self):
        # please overwrite
        assert True


class RevolveJoint(BaseJoint):
    def drive(self, angle: float):
        v = angle*self.axis
        a = self.origin.transform_direction(v)
        # a: rotation vector in world-space
        q = quaternion.from_rotation_vector(a)
        self.child.rotate_around(q, self.origin.position)
