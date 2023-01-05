import json
import numpy as np
from .geometry import create_points, create_cylinder, create_line

def read_skeleton(skelname):
    with open(skelname, 'r') as f:
        data = json.load(f)
    return data

CONFIG = {
    'body25': [[ 1,  0],
    [ 2,  1],
    [ 3,  2],
    [ 4,  3],
    [ 5,  1],
    [ 6,  5],
    [ 7,  6],
    [ 8,  1],
    [ 9,  8],
    [10,  9],
    [11, 10],
    [12,  8],
    [13, 12],
    [14, 13],
    [15,  0],
    [16,  0],
    [17, 15],
    [18, 16],
    [19, 14],
    [20, 19],
    [21, 14],
    [22, 11],
    [23, 22],
    [24, 11]
    ]
}

def build_skel(skel, skeltype):
    keypoints3d = np.array(skel['keypoints3d'])
    pid = skel['id']
    for k3d in keypoints3d:
        create_points(vid=pid, radius=0.025, center=k3d[:3])
    kintree = CONFIG[skeltype]
    for (i, j) in kintree:
        create_line(pid, 0.02, start=keypoints3d[i, :3], end=keypoints3d[j, :3])

def build_skeleton(skelname, skeltype):
    skeletons = read_skeleton(skelname)
    for skel in skeletons:
        build_skel(skel, skeltype)