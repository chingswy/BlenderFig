'''
  @ Date: 2024-12-24
  @ Author: Qing Shuai
  @ Description: Render a sequence of sampled frames from an animation file (FBX)
                 arranged from left to right along the x-axis as a single shot.
'''
import os
import numpy as np
import bpy
from myblender.geometry import (
    set_camera,
    build_plane,
    create_camera_blender_animated,
)

from myblender.setup import (
    setLight_sun,
    setLight_ambient,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)

from myblender.material import set_material_i, add_material


def find_armature_and_mesh(obj_names):
    """Find armature and mesh objects from a list of object names."""
    armature = None
    mesh_object = None
    mesh_object_list = []
    for obj_name in obj_names:
        obj = bpy.data.objects[obj_name]
        if obj.type == 'ARMATURE' or (obj.animation_data and obj.animation_data.action):
            armature = obj
        if obj.type == 'MESH' and mesh_object is None:
            mesh_object = obj
        if obj.type == 'MESH':
            mesh_object_list.append(obj)
    return armature, mesh_object, mesh_object_list




if __name__ == '__main__':
    # ${blender} -noaudio --python examples/render_smpl_shot.py -- test_baseline.fbx --num_frames 8 --material_index 0 --plane_size 20.0
    parser = get_parser()
    parser.add_argument('--camera', type=str, default=None)
    parser.add_argument('--no_axis_convert', action='store_true',
                        help='Disable Y-up to Z-up coordinate conversion for camera data')
    args = parse_args(parser)

    # Setup scene
    setup()

    # Load the FBX file
    fbx_path = args.path
    assert fbx_path.endswith('.fbx') or fbx_path.endswith('.FBX'), \
        f"Input file must be an FBX file, got: {fbx_path}"

    print(f"Loading FBX file: {fbx_path}")
    assert os.path.exists(fbx_path), fbx_path
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    assert os.path.exists(args.camera), args.camera
    camera_data = dict(np.load(args.camera))

    print(camera_data.keys())
    # dict_keys(['camera_RT', 'end_effector_vel', 'pred_j3d_glob'])

    # Visualize the dynamic camera
    # camera_RT is in world_to_camera format with shape (num_frames, 3, 4) or (num_frames, 4, 4)
    # By default, convert from Y-up (npz data) to Z-up (Blender after FBX import)
    if 'camera_RT' in camera_data:
        camera_RT = camera_data['camera_RT']
        print(f"Camera RT shape: {camera_RT.shape}")
        # Create animated camera visualization
        # convert_axis=True: convert Y-up to Z-up (needed when FBX was imported with default settings)
        convert_axis = not args.no_axis_convert
        camera_vis = create_camera_blender_animated(
            camera_RT, scale=1, pid=0, start_frame=0, convert_axis=convert_axis
        )
        print(f"Created camera visualization object: {camera_vis.name}")
