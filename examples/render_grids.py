'''
  @ Date: 2022-09-13 12:32:11
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-09-13 12:36:22
  @ FilePath: /EasyMocapPublic/scripts/blender/render_example.py
'''
# TODO: This scripts show how to use blender to render a cube
import numpy as np
import bpy
import os
from myblender.geometry import (
    set_camera,
    create_any_mesh,
    create_points
)
from myblender.color import color_jet

from myblender.setup import (
    add_sunlight,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)
from myblender.grid import plot_grids
from myblender.skeleton import read_skeleton, add_skeleton, update_skeleton

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--radius', type=float, default=0.01)
    parser.add_argument('--sphere_res', type=int, default=2)
    args = parse_args(parser)

    setup()
    set_camera(location=(3, 0, 2.5), center=(0, 0, 1), focal=50)
    add_sunlight(name='Light', location=(0., 0., 5.), rotation=(0., np.pi/12, 0))

    grids = np.load(args.path)
    grids = grids[:, :, :16]
    grids = grids.reshape(-1, 3)
    # Create a new mesh
    mesh = bpy.data.meshes.new("PointCloud")

    # Add vertices to the mesh
    mesh.from_pydata(grids, [], [])

    skeleton = read_skeleton('/Users/shuaiqing/nas/home/shuaiqing/Code/EasyPose/runs/A100/demo/511final/version_7/511-balance/000350.jpg.json')
    skeleton = np.array(skeleton['pred'])
    for i in range(skeleton.shape[0]):
        add_skeleton(skeleton[i], pid='000000', skeltype='panoptic15', mode='ellips')
    # Create a new object with the mesh
    obj = bpy.data.objects.new("PointCloud", mesh)

    # Link the object to the scene
    bpy.context.scene.collection.objects.link(obj)
    # for i, grid in enumerate(grids):
    #     create_points(0, radius=0.02, center=grid)

    # setup render
    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=args.num_samples,
        use_transparent_bg=False,
        use_denoising=args.denoising,
    )

    n_parallel = 1
    if args.out is not None:
        set_output_properties(bpy.context.scene, output_file_path=args.out, 
            res_x=args.res_x, res_y=args.res_y, 
            tile_x=args.res_x//n_parallel, tile_y=args.res_y, resolution_percentage=100,
            format='JPEG')
        bpy.ops.render.render(write_still=True, animation=False)
    if args.out_blend is not None:
        bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)