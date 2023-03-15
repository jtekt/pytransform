import numpy as np
import quaternion


class Transform():
    name: str
    parent = None  # Transform
    child = None  # Transform

    __position: np.ndarray
    __rotation: quaternion.quaternion
    __scale: np.ndarray

    def __init__(
        self,
        position=np.array([0, 0, 0]),
        rotation=np.quaternion(1, 0, 0, 0),
        scale=np.eye(3)
    ) -> None:
        self.name = ""
        self.__position = position
        self.__rotation = rotation
        self.__scale = scale
