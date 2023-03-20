
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from mpl_toolkits.mplot3d.axes3d import Axes3D

import pytransform as pytf
from pytransform.joint import RevolveJoint
from pytransform.tf import Transform


def main():

    mytf = Transform(
        position=np.array([0.0, 1.0, 0.0]),
        name='parent')

    child_tf = Transform(
        position=np.array((0.0, 3.0, 0.0)),
        name='child'
    )

    child_tf2 = Transform(
        position=np.array((0.0, 3.0, 0.0)),
        name='child2'
    )

    j = RevolveJoint(
        parent=mytf, child=child_tf,
        origin=Transform(mytf.position, name='hinge'),
        axis=np.array([0, 0, 1]),
        limit=(1, -1)
    )

    child_tf2.set_parent(mytf)

    print("\n\r".join(mytf.tree()))

    # print(j.origin.tree())


if __name__ == '__main__':
    main()
