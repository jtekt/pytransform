# urdf parser
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from . import joint
from . import quaternion_utils as quat
from .chain import Chain
from .tf import Transform


def _elm2link(elm: ET.Element):
    if elm.tag != "link":
        raise ValueError(f'tag must be link')

    return Transform(
        name=elm.attrib['name']
    )


def str2array(s: str, decoder=float, deliminator: str = ' '):
    return [decoder(x) for x in s.split(deliminator)]


def _elm2joint(elm: ET.Element, links: list[Transform]):
    if elm.tag != "joint":
        raise ValueError(f'tag must be joint')
    name = elm.attrib['name']
    type_name = elm.attrib['type']
    # parent--child
    parent_name = elm.find('parent').attrib['link']
    child_name = elm.find('child').attrib['link']

    candidate_p = [l for l in links if l.name == parent_name]
    if len(candidate_p) == 0:
        return
    candidate_c = [l for l in links if l.name == child_name]
    if len(candidate_c) == 0:
        return
    parent = candidate_p[0]
    child = candidate_c[0]
    # joint origin
    origin_rot = quat.identity()  # rotation offset from parent
    origin_pos = [0, 0, 0]  # offset from parent
    origin = elm.find('origin')
    if origin is not None:
        if 'rpy' in origin.attrib:
            rpy = str2array(origin.attrib['rpy'])
            origin_rot = quat.from_rpy(*rpy)
        if 'xyz' in origin.attrib:
            origin_pos = str2array(origin.attrib['xyz'])

    org = Transform(
        position=parent.position+origin_pos,
        rotation=origin_rot*parent.rotation,
        name=name
    )

    child.rotate_to(org.rotation)
    child.translate_to(org.position)

    #  joint axis
    axis = [1, 0, 0]
    a_item = elm.find('axis')
    if a_item is not None:
        axis = str2array(a_item.attrib['xyz'])

    # joint limitation
    upper = 1.0e9
    lower = -1.0e9

    lim_item = elm.find('limit')
    if lim_item is not None:
        if 'upper' in lim_item.attrib:
            upper = float(lim_item.attrib['upper'])
        if 'lower' in lim_item.attrib:
            lower = float(lim_item.attrib['lower'])

    joint_dict = {
        'fixed': joint.FixedJoint,
        'revolute': joint.RevoluteJoint,
        'continuous': joint.ContinuousJoint,
        'prismatic': joint.PrismaticJoint
    }

    if type_name in joint_dict:
        return joint_dict[type_name](
            parent, child,
            origin=org,
            axis=axis,
            limit=joint.Limitation(upper, lower)
        )
    else:
        raise ValueError(f'{type_name} is not valid')


def chain_from_urdf(filename: str):
    tree = ET.parse(filename)
    root = tree.getroot()
    robot_name = root.attrib['name']

    link_elms = root.findall('link')
    links = [_elm2link(l) for l in link_elms]

    joint_elms = root.findall('joint')
    joints = [
        _elm2joint(j, links) for j in joint_elms
    ]

    has_next = True

    return Chain(
        links=links,
        joints=joints,
        name=robot_name
    )
