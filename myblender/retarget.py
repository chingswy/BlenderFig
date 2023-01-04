import bpy
import numpy as np

from mathutils import Matrix, Vector, Quaternion, Euler

def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

EYE3 = np.eye(3)

MAPPING_MIXAMO_SMPL = {
    # left leg
    "mixamorig:LeftUpLeg":{
        'idx': 1,
        'transform': EYE3
    },
    "mixamorig:LeftLeg": {
        'idx': 4,
        'transform': EYE3
    },
    "mixamorig:LeftFoot": {
        'idx': 7,
        'transform': EYE3
    },
    # right leg
    "mixamorig:RightUpLeg":{
        'idx': 2,
        'transform': EYE3
    },
    "mixamorig:RightLeg": {
        'idx': 5,
        'transform': EYE3
    },
    "mixamorig:RightFoot": {
        'idx': 8,
        'transform': EYE3
    },
    # spine
    "mixamorig:Spine": {
        'idx': 3,
        'transform': EYE3
    },
    "mixamorig:Spine1": {
        'idx': 6,
        'transform': EYE3
    },
    "mixamorig:Spine2": {
        'idx': 9,
        'transform': EYE3
    },
    # neck
    "mixamorig:Neck": {
        'idx': 12,
        'transform': EYE3
    },
    "mixamorig:Head":{
        'idx': 15,
        'transform': EYE3
    },
    # left shoulder
    "mixamorig:LeftShoulder": {
        'idx': 13,
        'transform': EYE3
    },
    "mixamorig:LeftArm": {
        'idx': 16,
        'transform': EYE3
    },
    "mixamorig:LeftForeArm": {
        'idx': 18,
        'transform': EYE3
    },
    # right shoulder
    "mixamorig:RightShoulder": {
        'idx': 14,
        'transform': EYE3
    },
    "mixamorig:RightArm": {
        'idx': 17,
        'transform': EYE3
    },
    "mixamorig:RightForeArm": {
        'idx': 19,
        'transform': EYE3
    },
}

def map_bone_name(bone_name):
    return bone_name

def set_trans_by_center(Rot, Th, center):
    # 当做是绕root点进行旋转的
    return Th - center

def animate_smpl_motion(arm_ob, bones, data):
    scene = bpy.data.scenes['Scene']
    root_name = map_bone_name("mixamorig:Hips")
    scale = arm_ob.scale
    center = arm_ob.pose.bones[root_name].center * scale
    for frame in range(data['Th'].shape[0]):
        scene.frame_set(frame)
        Rot = Rodrigues(data['Rh'][frame])
        Th = data['Th'][frame]
        bones[root_name].location = set_trans_by_center(Rot, Th, center)/scale
        bones[root_name].rotation_quaternion = Matrix(Rot).to_quaternion()
        bones[root_name].keyframe_insert('location', frame=frame)
        bones[root_name].keyframe_insert('rotation_quaternion', frame=frame)
        for bone_name_, cfg in MAPPING_MIXAMO_SMPL.items():
            bone_name = map_bone_name(bone_name_)
            idx = cfg['idx']
            rot = Matrix(cfg['transform']) @ Matrix(Rodrigues(data['poses'][frame, 3*idx:3*idx+3]))
            bones[bone_name].rotation_quaternion = rot.to_quaternion()
            bones[bone_name].keyframe_insert('rotation_quaternion', frame=frame)
            bones[bone_name].keyframe_insert('location', frame=frame)

    bpy.context.scene.frame_end = data['Th'].shape[0]
