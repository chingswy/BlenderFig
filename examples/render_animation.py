import os
from glob import glob
from os.path import join

from myblender.geometry import (
    set_camera,
    build_plane,
    look_at
)
from myblender.setup import (
    add_sunlight,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)

import bpy
import numpy as np
from mathutils import Vector, Quaternion, Matrix
import json

XBOT_BONES = [
    'mixamorig:Hips', 
    'mixamorig:Spine', 
    'mixamorig:Spine1', 
    'mixamorig:Spine2', 
    'mixamorig:Neck', 
    'mixamorig:Head', 
    'mixamorig:HeadTop_End', 
    'mixamorig:LeftEye', 
    'mixamorig:RightEye', 
    'mixamorig:LeftShoulder', 
    'mixamorig:LeftArm', 
    'mixamorig:LeftForeArm', 
    'mixamorig:LeftHand', 
    'mixamorig:LeftHandThumb1', 
    'mixamorig:LeftHandThumb2', 
    'mixamorig:LeftHandThumb3', 
    'mixamorig:LeftHandThumb4', 
    'mixamorig:LeftHandIndex1', 
    'mixamorig:LeftHandIndex2', 
    'mixamorig:LeftHandIndex3', 
    'mixamorig:LeftHandIndex4', 
    'mixamorig:LeftHandMiddle1', 
    'mixamorig:LeftHandMiddle2', 
    'mixamorig:LeftHandMiddle3', 
    'mixamorig:LeftHandMiddle4', 
    'mixamorig:LeftHandRing1', 
    'mixamorig:LeftHandRing2', 
    'mixamorig:LeftHandRing3', 
    'mixamorig:LeftHandRing4', 
    'mixamorig:LeftHandPinky1', 
    'mixamorig:LeftHandPinky2', 
    'mixamorig:LeftHandPinky3', 
    'mixamorig:LeftHandPinky4', 
    'mixamorig:RightShoulder', 
    'mixamorig:RightArm', 
    'mixamorig:RightForeArm', 
    'mixamorig:RightHand', 
    'mixamorig:RightHandPinky1', 
    'mixamorig:RightHandPinky2', 
    'mixamorig:RightHandPinky3', 
    'mixamorig:RightHandPinky4', 
    'mixamorig:RightHandRing1', 
    'mixamorig:RightHandRing2', 
    'mixamorig:RightHandRing3', 
    'mixamorig:RightHandRing4', 
    'mixamorig:RightHandMiddle1', 'mixamorig:RightHandMiddle2', 'mixamorig:RightHandMiddle3', 'mixamorig:RightHandMiddle4', 'mixamorig:RightHandIndex1', 'mixamorig:RightHandIndex2', 'mixamorig:RightHandIndex3', 'mixamorig:RightHandIndex4', 'mixamorig:RightHandThumb1', 'mixamorig:RightHandThumb2', 'mixamorig:RightHandThumb3', 'mixamorig:RightHandThumb4', 
    'mixamorig:LeftUpLeg', 
    'mixamorig:LeftLeg', 
    'mixamorig:LeftFoot', 
    'mixamorig:LeftToeBase', 
    'mixamorig:LeftToe_End', 
    'mixamorig:RightUpLeg', 
    'mixamorig:RightLeg', 
    'mixamorig:RightFoot', 
    'mixamorig:RightToeBase', 
    'mixamorig:RightToe_End']

MAP_SMPL_XBOT = [
    "mixamorig:Hips",
    # left leg
    "mixamorig:LeftUpLeg",
    "mixamorig:RightUpLeg",
    "mixamorig:Spine",
    # 
    "mixamorig:LeftLeg",
    "mixamorig:RightLeg",
    "mixamorig:Spine1",
    # 
    "mixamorig:LeftFoot",
    "mixamorig:RightFoot",
    "mixamorig:Spine2",
    # 
    "mixamorig:LeftToeBase",
    "mixamorig:RightToeBase",
    "mixamorig:Neck",
    # 13
    "mixamorig:LeftShoulder",
    "mixamorig:RightShoulder",
    "mixamorig:Head",
    # 16
    "mixamorig:LeftArm",
    "mixamorig:RightArm",
    # # right leg
    # # spine
    # # neck
    # 18
    "mixamorig:LeftForeArm",
    "mixamorig:RightForeArm",
    # 20, 21
    "mixamorig:LeftHand",
    "mixamorig:RightHand",
    # "",
    # "",
]

def read_smpl(filename='/tmp/smpl.json'):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def Rodrigues(rotvec):
    theta = np.linalg.norm(rotvec)
    r = (rotvec/theta).reshape(3, 1) if theta > 0. else rotvec
    cost = np.cos(theta)
    mat = np.asarray([[0, -r[2], r[1]],
                      [r[2], 0, -r[0]],
                      [-r[1], r[0], 0]])
    return(cost*np.eye(3) + (1-cost)*r.dot(r.T) + np.sin(theta)*mat)

def animate_by_smpl(param, bones, frame):
    if args.retarget:
        root_name = 'mixamorig:Hips'
        scale = 100.
        zoffset = 0
    else:
        root_name = 'm_avg_root'
        scale = 1.
        zoffset = 1.
    scene = bpy.context.scene
    scene.frame_set(frame)
    trans = param['Th'][0]
    bones[root_name].location = Vector((scale*trans[0], scale*trans[1], scale*(trans[2])-zoffset))
    bones[root_name].keyframe_insert('location', frame=frame)
    bone_rotation = Rodrigues(np.array(param['Rh'][0]))
    bone_rotation = Matrix(bone_rotation).to_quaternion()   
    # bones[root_name].rotation_quaternion = bone_rotation
    bones[root_name].keyframe_insert('rotation_quaternion', frame=frame)
    for i in range(1, len(MAP_SMPL_XBOT)):
        break
        rvec = np.array(param['poses'][0][3*i-3:3*i])
        if args.retarget:
            bone_name = MAP_SMPL_XBOT[i]
            if bone_name in ['mixamorig:LeftUpLeg', 'mixamorig:RightUpLeg', 'mixamorig:LeftLeg', 'mixamorig:RightLeg', 'mixamorig:LeftFoot', 'mixamorig:RightFoot', 'mixamorig:LeftToeBase', 'mixamorig:RightToeBase']:
                rvec[1] *= -1
                rvec[2] *= -1
            if bone_name in ['mixamorig:RightShoulder', 'mixamorig:RightArm', 'mixamorig:RightForeArm', 'mixamorig:RightHand']:
                rvec[[0,1]] = rvec[[1,0]]
            if bone_name in ['mixamorig:LeftShoulder', 'mixamorig:LeftArm', 'mixamorig:LeftForeArm', 'mixamorig:LeftHand']:
                rvec[[0,1]] = rvec[[1,0]]
                rvec[2] *= -1
        else:
            bone_name = list(bones.keys())[i]
        bone_rotation = Rodrigues(rvec)
        bone_rotation = Matrix(bone_rotation).to_quaternion()
        bones[bone_name].rotation_quaternion = bone_rotation
        bones[bone_name].keyframe_insert('rotation_quaternion', frame=frame)

def run():
    params = read_smpl()
    target_model = 'Armature'
    character = bpy.data.objects[target_model]
    bones = character.pose.bones
    animate_by_smpl(params[0], bones, 0)

def load_smpl_from_dir(dirname):
    filenames = sorted(glob(os.path.join(dirname, '*.json')))
    bpy.context.scene.frame_end = len(filenames)
    target_model = 'Armature'
    character = bpy.data.objects[target_model]
    bones = character.pose.bones
    for frame, filename in enumerate(filenames):
        params = read_smpl(join(dirname, filename))
        # TODO: 只兼容一个人
        if isinstance(params, dict):
            params = params['annots']
        animate_by_smpl(params[0], bones, frame)


def get_calibration_matrix_K_from_blender(mode='simple'):

    scene = bpy.context.scene

    scale = scene.render.resolution_percentage / 100
    width = scene.render.resolution_x * scale # px
    height = scene.render.resolution_y * scale # px

    camdata = scene.camera.data

    if mode == 'simple':

        aspect_ratio = width / height
        K = np.zeros((3,3), dtype=np.float32)
        K[0][0] = width / 2 / np.tan(camdata.angle / 2)
        K[1][1] = height / 2. / np.tan(camdata.angle / 2) * aspect_ratio
        K[0][2] = width / 2.
        K[1][2] = height / 2.
        K[2][2] = 1.
        K.transpose()
    
    if mode == 'complete':

        focal = camdata.lens # mm
        sensor_width = camdata.sensor_width # mm
        sensor_height = camdata.sensor_height # mm
        pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

        if (camdata.sensor_fit == 'VERTICAL'):
            # the sensor height is fixed (sensor fit is horizontal), 
            # the sensor width is effectively changed with the pixel aspect ratio
            s_u = width / sensor_width / pixel_aspect_ratio 
            s_v = height / sensor_height
        else: # 'HORIZONTAL' and 'AUTO'
            # the sensor width is fixed (sensor fit is horizontal), 
            # the sensor height is effectively changed with the pixel aspect ratio
            pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
            s_u = width / sensor_width
            s_v = height * pixel_aspect_ratio / sensor_height

        # parameters of intrinsic calibration matrix K
        alpha_u = focal * s_u
        alpha_v = focal * s_v
        u_0 = width / 2
        v_0 = height / 2
        skew = 0 # only use rectangular pixels

        K = np.array([
            [alpha_u,    skew, u_0],
            [      0, alpha_v, v_0],
            [      0,       0,   1]
        ], dtype=np.float32)
    
    return K

def set_intrinsic(K, camera, sensor_width=1.0):
    scene = bpy.context.scene
    # Intrinsic
    f_x = K[0, 0]
    f_y = K[1, 1]
    c_x = K[0, 2]
    image_width = c_x * 2  # principal point x assumed at the center
    cam = camera.data
    cam.name = 'CamFrom3x4P'
    cam.type = 'PERSP'
    cam.lens = f_x / image_width * sensor_width
    cam.lens_unit = 'MILLIMETERS'
    cam.sensor_width = sensor_width
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = f_y / f_x
    Knew = get_calibration_matrix_K_from_blender(mode='complete')
    print(K)
    print(Knew)

def set_extrinsic(R_world2cv, T_world2cv, camera):
    R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
    R_cv2world = R_world2cv.T
    rotation = Matrix(R_cv2world.tolist()) @ R_bcam2cv
    location = -R_cv2world @ T_world2cv
    camera.location = location
    camera.matrix_world = Matrix.Translation(location) @ rotation.to_4x4()

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--source", type=str)
    parser.add_argument("--target", type=str, help="should be a fbx file for now", nargs='+')
    parser.add_argument("--zoffset", type=float, help="offset for z axis")
    parser.add_argument('--retarget', action='store_true')
    args = parse_args(parser)
    print(args)

    setup()
    add_sunlight(name='Light', location=(0., 0., 5.), rotation=(0., np.pi/12, 0))
    set_camera(location=(3, 0, 2.5), center=(0, 0, 1), focal=30)
    K = np.eye(3)
    K[0, 0] = 1080*1.2
    K[1, 1] = 1080*1.2
    K[0, 2] = 1920/2
    K[1, 2] = 1080/2
    R = np.eye(3)
    T = np.zeros((3, 1))
    set_intrinsic(K, bpy.data.objects['Camera'])
    set_extrinsic(R, T, bpy.data.objects['Camera'])


    for target in args.target:
        if 'SMPL_maya' in target:
            scale = 100.
            bpy.ops.import_scene.fbx(
                filepath=target,
                axis_forward="Y",
                axis_up="Z",
                global_scale=scale
            )
        else:
            scale = 1.
            bpy.ops.import_scene.fbx(
                filepath=target,
                use_manual_orientation=True,
                use_anim=False,
                axis_forward="Y",
                axis_up="Z",
                automatic_bone_orientation=True,
                global_scale=scale
            )
        obj_names = [o.name for o in bpy.context.selected_objects]
        ## e.g., ['Armature', 'Beta_Joints', 'Beta_Surface'] for `xbot.fbx`
        target_model = obj_names[0]

        character = bpy.data.objects[target_model]
        character.location = Vector((0, -1, 2))
        armature = bpy.data.armatures[target_model]
        armature.animation_data_clear()

        bones = character.pose.bones
        bones_name = list(bones.keys())
        print('Bones: ', bones_name)
        scene = bpy.context.scene
        insert_interval = 1
        load_smpl_from_dir(args.path)
    
    nFrames = bpy.context.scene.frame_end
    camera = bpy.data.objects['Camera']
    # camera.animation_data_clear()
    # camera.keyframe_insert(
    #     "location", frame=0
    # )
    # look_at(camera, Vector((0, 0, 5)))
    # camera.location = camera.location + Vector((5, 0, 0))
    # camera.keyframe_insert(
    #     "location", frame=nFrames-1
    # )
    # look_at(camera, Vector((0, 0, 5)))
    
    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=args.num_samples,
        use_transparent_bg=False,
        use_denoising=args.denoising,
    )

    set_output_properties(bpy.context.scene, output_file_path=args.out, 
        res_x=args.res_x, res_y=args.res_y, 
        tile_x=args.res_x, tile_y=args.res_y, resolution_percentage=100,
        format='JPEG')
    
    if not args.debug:
        bpy.ops.render.render(write_still=True, animation=True)