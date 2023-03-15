import numpy as np
import quaternion


class Transform():
    name: str
    _position: np.ndarray
    _rotation: quaternion.quaternion
    _scale: np.ndarray

    def __init__(self) -> None:
        self.name = ""
        self._position = np.array([0, 0, 0])
        self._rotation = np.quaternion(1, 0, 0, 0)
        self._scale = np.eye(3)
