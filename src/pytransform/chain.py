from __future__ import annotations

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
