import json
import numpy as np
import bpy
from .geometry import create_points, create_cylinder, create_line, create_ellipsold, look_at

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
    ],
    'body15': [[ 1,  0],
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
    [14, 13]
    ],
    'panoptic15':[
        [0, 1],
        [0, 2],
        [0, 3],
        [3, 4],
        [4, 5],
        [0, 9],
        [9, 10],
        [10, 11],
        [2, 6],
        [2, 12],
        [6, 7],
        [7, 8],
        [12, 13],
        [13, 14]]
}

def add_skeleton(keypoints3d, pid, skeltype, mode='line', color=None, frame=None):
    points = []
    for k3d in keypoints3d:
        if len(k3d) == 4 and k3d[3] < 0.01:
            points.append(None)
            continue
        obj = create_points(vid=pid, radius=0.025, center=k3d[:3])
        bpy.ops.object.shade_smooth()
        points.append(obj)
    limbs = []
    kintree = CONFIG[skeltype]
    for (i, j) in kintree:
        if keypoints3d.shape[1] == 4 and (keypoints3d[i, 3] < 0.01 or keypoints3d[j, 3] < 0.01):
            limbs.append(None)
            continue
        if mode == 'line':
            obj = create_line(pid, 0.02, start=keypoints3d[i, :3], end=keypoints3d[j, :3])
        else:
            obj = create_ellipsold(pid, 0.02, start=keypoints3d[i, :3], end=keypoints3d[j, :3])
        bpy.ops.object.shade_smooth()
        
        limbs.append(obj)
    return points, limbs

def update_skeleton(keypoints3d, skeltype, points, limbs, frame):
    # TODO: 修正一下骨长；以第一帧的骨长为准

    for i in range(keypoints3d.shape[0]):
        points[i].location = keypoints3d[i, :3]
        points[i].keyframe_insert('location', frame=frame)
    kintree = CONFIG[skeltype]
    for ilimb, (i, j) in enumerate(kintree):
        obj = limbs[ilimb]
        start, end = keypoints3d[i, :3], keypoints3d[j, :3]
        length = np.linalg.norm(end - start)
        obj.location = (keypoints3d[i, :3] + keypoints3d[j, :3]) / 2
        radius = obj.scale[0]
        scale = (radius, radius, length/2)
        obj.scale = scale
        obj.keyframe_insert('scale', frame=frame)
        look_at(obj, keypoints3d[j, :3])
        obj.keyframe_insert('location', frame=frame)
        obj.keyframe_insert('rotation_euler', frame=frame)


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