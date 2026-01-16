import sys
import argparse
import os
import math
import bpy

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, default=None, nargs='+')
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

def zero_xy_translation(armature):
    """将动作数据中的xy平面位移设置为0（针对Pelvis根骨骼）"""
    if not armature or not armature.animation_data or not armature.animation_data.action:
        print("No animation data found to zero out XY translation")
        return

    action = armature.animation_data.action

    # 先打印所有的fcurves用于调试
    print("=== All fcurves in action ===")
    for fcurve in action.fcurves:
        print(f"  {fcurve.data_path}[{fcurve.array_index}], keyframes: {len(fcurve.keyframe_points)}")
    print("=============================")

    # 遍历所有fcurves，找到位置相关的曲线
    for fcurve in action.fcurves:
        # 检查是否是根骨骼(Pelvis)的location动画
        # data_path通常是 "pose.bones[\"Pelvis\"].location" 或 "location"
        # array_index: 0=X, 1=Y, 2=Z
        is_root_location = (
            fcurve.data_path == 'location' or  # armature本身的位移
            ('Pelvis' in fcurve.data_path and 'location' in fcurve.data_path)  # Pelvis骨骼的位移
        )

        if is_root_location:
            # 只处理X和Y分量 (array_index 0 和 1)
            if fcurve.array_index in [0, 1]:
                num_keyframes = len(fcurve.keyframe_points)
                print(f"Zeroing {num_keyframes} keyframes for: {fcurve.data_path}[{fcurve.array_index}]")
                # 将所有关键帧的值设置为0
                for keyframe in fcurve.keyframe_points:
                    keyframe.co[1] = 0.0  # co[1]是关键帧的值
                    keyframe.handle_left[1] = 0.0
                    keyframe.handle_right[1] = 0.0
                # 更新fcurve确保修改生效
                fcurve.update()

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--zero', action='store_true')
    parser.add_argument('--no_trans', action='store_true', help='将所有动作数据的xy平面位移设置为0')
    args = parse_args(parser)

    paths = sorted(args.path)
    num_items = len(paths)
    # 计算接近正方形的网格尺寸
    cols = max(2, math.ceil(math.sqrt(num_items)))
    if cols % 2 != 0:
        cols += 1
    rows = max(1, math.ceil(num_items / cols))
    spacing = 2.0  # 网格间距，可根据模型大小调整

    for index, fbx_path in enumerate(paths):
        print(fbx_path)
        assert os.path.exists(fbx_path), fbx_path
        try:
            bpy.ops.import_scene.fbx(filepath=fbx_path)
        except Exception as e:
            print(e)
            continue

        # 修改场景的fps，start frame end frame，根据fbx文件的fps和帧数来计算
        obj_names = [o.name for o in bpy.context.selected_objects]
        armature, mesh_object, mesh_object_list = find_armature_and_mesh(obj_names)

        # 将所有导入对象的名字改成文件名的basename
        file_basename = os.path.splitext(os.path.basename(fbx_path))[0]
        for i, obj_name in enumerate(obj_names):
            obj = bpy.data.objects[obj_name]
            if len(obj_names) == 1:
                obj.name = file_basename
            else:
                obj.name = f"{file_basename}_{i}"

        # 如果启用no_trans，将xy平面位移设置为0
        if args.no_trans:
            zero_xy_translation(armature)

        # 设置位置为接近正方形的矩形网格分布
        if not args.zero:
            row = index // cols
            col = index % cols
            armature.location.x = col * spacing
            armature.location.y = -row * spacing
            set_scene_frame_range(armature)