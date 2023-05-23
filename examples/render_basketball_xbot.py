import bpy
import os
from glob import glob
from os.path import join
import numpy as np
from mathutils import Vector, Quaternion, Matrix
from myblender.geometry import (
    set_camera,
    build_plane,
    create_plane_blender,
    look_at
)
from myblender.camera import set_extrinsic, set_intrinsic
from myblender.animation import read_smpl, animate_by_smpl
from myblender.setup import (
    add_sunlight,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)

def set_camera_green(camera):
    K =[13365.842100, 0.000000, 1266.274010, 0.000000, 13353.006300, 1029.775970, 0.000000, 0.000000, 1.000000]
    K = np.array(K).reshape(3, 3)
    set_intrinsic(K, camera, 2448, 2048)
    R = [0.021391, 0.999674, -0.013928, 0.160876, -0.017191, -0.986825, -0.986743, 0.018868, -0.161191]
    T = [1.414816, -0.064123, 30.749459]
    R = np.array(R).reshape(3, 3)
    T = np.array(T).reshape(3, 1)
    set_extrinsic(R, T, camera)

color_table = [
    (94/255, 124/255, 226/255), # 青色
    (255/255, 200/255, 87/255), # yellow
    (74/255.,  189/255.,  172/255.), # green
    (8/255, 76/255, 97/255), # blue
    (219/255, 58/255, 52/255), # red
    (77/255, 40/255, 49/255), # brown
    '114B5F',
    'D89D6A',
    'A0A4B8',
    '2EC0F9',
    '30332E', # 淡黑色
    'F2D1C9', # 淡粉色
]

for i, color in enumerate(color_table):
    if isinstance(color, str):
        color_table[i] = tuple(map(lambda x: int(x, 16)/255., [color[:2], color[2:4], color[4:]]))
    color_table[i] = (*color_table[i], 1.0)

if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--source", type=str)
    parser.add_argument("--zoffset", type=float, help="offset for z axis")
    parser.add_argument('--retarget', action='store_true')
    args = parse_args(parser)
    print(args)

    setup()
    add_sunlight(name='Light0', location=(2., 5., 5.), rotation=(-np.pi/4, 0, 0), strength=1.)
    add_sunlight(name='Light1', location=(2., -5., 5.), rotation=(np.pi/4, 0, 0), strength=1.)

    camera = set_camera(location=(3, 0, 2.5), center=(0, 0, 1), focal=30)
    set_camera_green(camera)

    pids = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    mapname = {
        # 0: 'assets/humanfbx/Ch13_nonPBR.fbx',
        # 1: 'assets/humanfbx/Ch28_nonPBR.fbx',
        # 2: 'assets/humanfbx/Ch41_nonPBR.fbx',
        # 3: 'assets/humanfbx/Ch22_nonPBR.fbx',
        # 4: 'assets/humanfbx/Ch33_nonPBR.fbx',
        # 5: 'assets/humanfbx/Ch46_nonPBR.fbx',
        # 6: 'assets/humanfbx/Remy.fbx',
        # 7: 'assets/humanfbx/Kachujin G Rosales.fbx',
        8: 'assets/xbot.fbx',
    }
    # 预先load所有的模型进来
    models = {}
    for pid in pids:
        target = mapname.get(pid, 'assets/xbot.fbx')
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
        # 设置joints和surface的scale
        target_model = obj_names[0]
        for obj_name in obj_names[1:]:
            matname = list(bpy.data.objects[obj_name].material_slots.keys())[0]
            bpy.data.materials[matname].node_tree.nodes["Principled BSDF"].inputs[0].default_value = color_table[pid]
        character = bpy.data.objects[target_model]
        character.location = Vector((0, -0.5, 0))
        armature = bpy.data.armatures[target_model]
        armature.animation_data_clear()
        models[pid] = target_model
    
    filenames = sorted(glob(os.path.join(args.path, '*.json')))
    bpy.context.scene.frame_end = len(filenames)

    args.res_x = 2448
    args.res_y = 2048

    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=args.num_samples,
        use_transparent_bg=True,
        use_denoising=args.denoising,
    )

    scene = bpy.context.scene
    offset = [0, -0.5, -0.2]
    for frame, filename in enumerate(filenames):
        scene.frame_set(frame)
        print('Loading frames: ', frame)
        params = read_smpl(join(args.path, filename))
        # TODO: 只兼容一个人
        if isinstance(params, dict):
            params = params['annots']
        for param in params:
            pid = param['id']
            character = bpy.data.objects[models[pid]]
            bones = character.pose.bones
            animate_by_smpl(param, bones, frame, offset=offset)

    nFrames = bpy.context.scene.frame_end
    camera = bpy.data.objects['Camera']
    
    # 创建一个地面用来制造阴影
    plane = create_plane_blender((7, 0, 0.0), size=15)
    plane.hide_viewport = True
    plane.is_shadow_catcher = True

    if not args.debug and args.out is not None:
        set_output_properties(bpy.context.scene, 
            output_file_path=args.out,
            res_x=args.res_x, res_y=args.res_y, 
            tile_x=args.res_x, tile_y=args.res_y, resolution_percentage=100,
            format='PNG')
        
        # bpy.ops.render.render(write_still=True, animation=True)
        bpy.context.scene.frame_set(40)
        bpy.ops.render.render(write_still=True, animation=False)
