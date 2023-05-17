# 这个代码用于使用一个金字塔框架来可视化相机的位置和方向
import numpy as np
from os.path import join
import bpy
from myblender.geometry import (
    set_camera,
    build_plane
)

from myblender.camera_file import read_camera
from myblender.setup import (
    add_sunlight,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)
from myblender.skeleton import read_skeleton, add_skeleton, update_skeleton

from myblender.geometry import create_camera_blender

if __name__ == '__main__':
    parser = get_parser()

    args = parse_args(parser)

    setup(rgb=(1,1,1,0))
    set_camera(location=(0, 5, 7), center=(0, 0, 0), focal=30)
    add_sunlight(name='Light', location=(0., 0., 5.), rotation=(0., 0, 0), strength=5)
    plane = build_plane(translation=(0, 0, 0), plane_size=5)
    args.res_x = args.res_y = 512

    config = {
        'hi4d': {
            'root': '/Users/shuaiqing/nas/home/shuaiqing/datasets/HI4D_easymocap/pair32_pose32',
            'color': 'FF0033'
        },
        'chi3d': {
            'root': '/Users/shuaiqing/nas/home/shuaiqing/datasets/chi3d_easymocap/s03_Grab 1',
            'color': '000000'
        },
        'panoptic': {
            'root': '/Users/shuaiqing/nas/home/shuaiqing/datasets/panoptic/160224_haggling1',
            'color': '003366'
        },
        # 'zjumocap': {
        #     'root': '/Users/shuaiqing/nas/ZJUMoCap/DeepMocap/511-balance',
        #     'color': [0., 1., 0., 1.]
        # },
        # 'basketball': {
        #     'root': '/Users/shuaiqing/nas/home/shuaiqing/extern/greenlint',
        #     'color': [1., 0., 0., 1.]
        # }
    }

    k3dname = join(config['panoptic']['root'], 'output-keypoints3d', 'keypoints3d', '001000.json')
    skeleton = read_skeleton(k3dname)
    for d in skeleton:
        add_skeleton(np.array(d['keypoints3d']), pid='000000', skeltype='body25', mode='ellips')
    for name, cfg in config.items():
        root = cfg['root']
        color = cfg['color']
        cameras = read_camera(join(root, 'intri.yml'), join(root, 'extri.yml'))

        for cam, camera in cameras.items():
            R = camera['R']
            T = camera['T']
            create_camera_blender(R, T, scale=0.3, pid=color)


    n_parallel = 1
    if args.out.endswith('.jpg'):
        format = 'JPEG'
    elif args.out.endswith('.png'):
        format = 'PNG'
    # setup render
    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=args.num_samples,
        use_transparent_bg=format=='PNG',
        use_denoising=args.denoising,
    )
    set_output_properties(bpy.context.scene, output_file_path=args.out, 
        res_x=args.res_x, res_y=args.res_y, 
        tile_x=args.res_x//n_parallel, tile_y=args.res_y, resolution_percentage=100,
        format=format)
    bpy.ops.render.render(write_still=True, animation=False)
    # if args.out_blend is not None:
    #     bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)