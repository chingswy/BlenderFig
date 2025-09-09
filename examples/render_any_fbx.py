import sys
import argparse
import os
import bpy

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default=None)
    return parser

def parse_args(parser):
    if '--' in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    else:
        args = parser.parse_args(['debug'])
    return args

def find_armature_and_mesh(obj_names):
    # Find the armature (assuming it's the first object or has animation data)
    armature = None
    mesh_object = None
    mesh_object_list = []
    for obj_name in obj_names:
        obj = bpy.data.objects[obj_name]
        print(obj_name, obj.type)
        if obj.type == 'ARMATURE' or (obj.animation_data and obj.animation_data.action):
            armature = obj
        if obj.type == 'MESH' and mesh_object is None:
            mesh_object = obj
        if obj.type == 'MESH':
            mesh_object_list.append(obj)

    return armature, mesh_object, mesh_object_list

def set_scene_frame_range(armature):
    if armature and armature.animation_data and armature.animation_data.action:
        # 打印fps
        # Get animation frame range from the action
        action = armature.animation_data.action
        # Action doesn't have fps attribute, use scene's fps instead
        print(f"FPS: {bpy.context.scene.render.fps}")
        frame_start = int(action.frame_range[0])
        frame_end = int(action.frame_range[1])

        # Set scene frame range to match the animation
        bpy.context.scene.frame_start = frame_start
        bpy.context.scene.frame_end = frame_end

        print(f"Animation frames set: {frame_start} to {frame_end}")
    else:
        print("No animation data found in the imported FBX")

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--video', type=str, default=None)
    args = parse_args(parser)

    fbx_path = args.path
    assert os.path.exists(fbx_path), fbx_path
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    # 修改场景的fps，start frame end frame，根据fbx文件的fps和帧数来计算
    obj_names = [o.name for o in bpy.context.selected_objects]
    armature, mesh_object, mesh_object_list = find_armature_and_mesh(obj_names)
    set_scene_frame_range(armature)


