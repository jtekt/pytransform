
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from mpl_toolkits.mplot3d.axes3d import Axes3D

import pytransform as pytf
from pytransform.joint import RevolveJoint
from pytransform.tf import Transform


def main():

    root_tf = Transform(
        position=np.array([0.0, 1.0, 0.0]),
        name='root')

    namihei = Transform(
        position=np.array((0.0, 3.0, 0.0)),
        name='namihei'
    )
    namihei.set_parent(root_tf)
    umihei = Transform(
        position=np.array((0.0, 3.0, 0.0)),
        name='umihei'
    )
    umihei.set_parent(root_tf)

    sazae = Transform(name='sazae')
    fune = Transform(name='fune')
    j = RevolveJoint(
        parent=namihei, child=sazae,
        origin=fune
    )

    Transform(name='katsuo').set_parent(fune)
    Transform(name='wakame').set_parent(fune)
    RevolveJoint(
        parent=sazae, child=Transform(name='tara'),
        origin=Transform(name='masuo')
    )

    Transform(name='?').set_parent(umihei)

    print(root_tf.tree())

    # print(j.origin.tree())


if __name__ == '__main__':
    main()
