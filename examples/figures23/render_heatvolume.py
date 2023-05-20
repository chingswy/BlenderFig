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
import cv2
import bpy
from myblender.color import color_jet
from myblender.geometry import (
    set_camera,
    create_bbox3d
)

from myblender.setup import (
    add_sunlight,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)
from myblender.geometry import create_plane, create_points, create_camera_blender, add_material
from myblender.skeleton import add_skeleton
from myblender.material import setMat_plastic, colorObj, set_material_i, build_pbr_nodes

def create_small_cube(color, scale, location, alpha, matmode):
    bpy.ops.mesh.primitive_cube_add(size=2, enter_editmode=False, align='WORLD')
    obj = bpy.context.object
    obj.rotation_euler = (0, 0, 0)
    if matmode == 'plastic':
        setMat_plastic(obj, colorObj(color, B=0.1), roughness=0.5, alpha=alpha)
    else:
        name = obj.name
        matname = "Material_{}".format(name)
        mat = add_material(matname, use_nodes=True, make_node_tree_empty=False)
        obj.data.materials.append(bpy.data.materials[matname])
        build_pbr_nodes(mat.node_tree, base_color=color, alpha=alpha)
    obj.scale = (scale/2,scale/2, scale/2)
    obj.location = location

def point_to_line_segment_distance(point, endpoint1, endpoint2):
    # points: (-1, 3)
    # endpoint1: (1, 3)
    # endpoint2: (1, 3)
    ba = endpoint2 - endpoint1
    pa = point - endpoint1
    t = (pa*ba).sum(axis=-1, keepdims=True) / (1e-5 + ba*ba).sum(axis=-1, keepdims=True)
    t = np.clip(t, 0, 1)
    h = endpoint1 + t * ba
    return np.linalg.norm(point - h, axis=-1)

def plot_input_volume(position, camera0, camera1, points_all, thres=0.1, sigma=0.1):
    dist0_to_line0 = point_to_line_segment_distance(position, camera0, points_all[0] + (points_all[0] - camera0))
    dist1_to_line0 = point_to_line_segment_distance(position, camera0, points_all[1] + (points_all[1] - camera0))
    dist0_to_line1 = point_to_line_segment_distance(position, camera1, points_all[0] + (points_all[0] - camera1))
    dist1_to_line1 = point_to_line_segment_distance(position, camera1, points_all[1] + (points_all[1] - camera1))
    

    dist = np.min(np.stack((dist0_to_line0, dist1_to_line0, dist0_to_line1, dist1_to_line1), axis=-1), axis=-1)
    aff = np.exp(-dist**2/(2*sigma**2))

    valid = (aff > thres) & (position[:, 2] < 0.01)

    position, aff = position[valid], aff[valid]
    color = (aff* 255).astype(np.int)
    color = color_jet[color]
    for i in range(position.shape[0]):
        print('Add {}/{}'.format(i, position.shape[0]))
        create_small_cube(color[i], scale=step, location=position[i], alpha=aff[i], matmode='plastic')

def create_camera(center0, center1):
    R0 = cv2.Rodrigues(np.array([0, -np.pi/2, 0.]).reshape(3, 1))[0]
    R0 = cv2.Rodrigues(np.array([0, 0., np.pi/4]).reshape(3, 1))[0] @ R0
    R0 = R0.T
    T0 = - R0 @ center0
    R1 = cv2.Rodrigues(np.array([0, np.pi/2, 0.]).reshape(3, 1))[0]
    R1 = cv2.Rodrigues(np.array([0, 0., -np.pi/4]).reshape(3, 1))[0] @ R1
    R1 = R1.T
    T1 = - R1 @ center1
    
    create_camera_blender(R0, T0, pid='000000')
    create_camera_blender(R1, T1, pid='000000')

def render():
    # setup render
    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=args.num_samples,
        use_transparent_bg=True,
        use_denoising=True,
    )

    n_parallel = 1
    set_output_properties(bpy.context.scene, output_file_path=args.out, 
        res_x=args.res_x, res_y=args.res_y, 
        tile_x=args.res_x//n_parallel, tile_y=args.res_y, resolution_percentage=100,
        format='PNG')
    bpy.ops.render.render(write_still=True, animation=False)
    if args.out_blend is not None:
        bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)

def plot_max_peak(position, points_all, thres=0.1, sigma=0.1, out='output/heatmap_peak.png'):
    dist = np.linalg.norm(position[:, None] - points_all[None], axis=-1).min(axis=1)
    aff = np.exp(-dist**2/(2*sigma**2))

    valid = (aff > thres)

    position, aff = position[valid], aff[valid]
    aff = aff / aff.max()
    color = (aff* 255).astype(np.int)
    color = color_jet[color]
    for i in range(position.shape[0]):
        print('Add {}/{}'.format(i, position.shape[0]))
        create_small_cube(color[i], scale=step, location=position[i], alpha=aff[i]**2, matmode='plastic')
    args.out = out


if __name__ == '__main__':
    parser = get_parser()
    args = parse_args(parser)

    setup()
    set_camera(location=(0, 4, 2), center=(0, 0, 0), focal=30)
    add_sunlight(name='Light', location=(0., 0., 5.), rotation=(0., np.pi/12, 0))

    N = 32
    step = 2/N
    create_bbox3d(pid=(94/255, 124/255, 226/255,1), scale=(1+step/2, 1+step/2, 1+step/2))
    sigma = 0.05
    thres = 0.2

    points0 = [-0.5, 0., 0.]
    points1 = [0.4, 0., 0.]
    camera0 = np.array([1.5, 1.5, 0.])
    camera1 = np.array([-1.5, 1.5, 0.])

    points_all = np.array([points0, points1])
    center0 = np.array(camera0).reshape(3, 1)
    center1 = np.array(camera1).reshape(3, 1)
    xcenter = np.linspace(-1., 1., N + 1)

    position = np.stack(np.meshgrid(xcenter, xcenter, xcenter), axis=-1).reshape(-1, 3)
    # create_camera(center0, center1)
    # plot_input_volume(xcenter, camera0, camera1, points_all, thres=thres, sigma=sigma)
    # plot_max_peak(position, points_all, thres=0.5, sigma=0.1)
    # plot_max_peak(position, points_all[:1], thres=0.5, sigma=0.1, out='output/heatmap_peak_1.png')
    kpts = np.load('output/keypoints.npy')[0]
    kpts[:, :3] = kpts[:, :3] - kpts[[2], :3]
    kpts[:, :3] = kpts[:, :3] @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    add_skeleton(kpts, pid=0, skeltype='panoptic15', mode='ellips')
    plot_max_peak(position, kpts[6, :3], thres=0.5, sigma=0.1, out='output/heatmap_results.png')
    render()