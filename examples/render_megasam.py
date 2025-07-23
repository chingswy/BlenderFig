'''
  @ Date: 2022-09-13 12:32:11
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-09-13 12:36:22
  @ FilePath: /EasyMocapPublic/scripts/blender/render_example.py
'''

import bpy
import trimesh
import os

import numpy as np
from myblender.geometry import (
    set_camera,
    build_plane
)

from myblender.setup import (
    add_sunlight,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
)
from myblender.geometry import create_plane, create_points
from myblender.camera import set_extrinsic, set_intrinsic

def depth_to_xyz(depth, K, cam_c2w):
    H, W = depth.shape
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v, indexing='xy')  # 像素坐标网格 (H, W)
    
    # 计算归一化相机坐标 (乘以深度前)
    x_norm = (uu - K[0, 2]) / K[0, 0]  # x方向归一化坐标 (H, W)
    y_norm = (vv - K[1, 2]) / K[1, 1]  # y方向归一化坐标 (H, W)
    
    # 直接构建4D齐次坐标数组 (H, W, 4)
    # 前三维是相机坐标系下的点坐标，第四维是齐次坐标分量
    points_cam = np.empty((H, W, 4), dtype=depth.dtype)
    points_cam[..., 0] = x_norm  # X_cam = x_norm * Z_cam
    points_cam[..., 1] = y_norm  # Y_cam = y_norm * Z_cam
    points_cam[..., 2] = depth           # Z_cam
    points_cam[..., 3] = 1.0             # 齐次分量
    
    # 转换到世界坐标系 (同时处理所有点)
    points_cam_flat = points_cam.reshape(-1, 4).T  # 转为(4, H*W)
    points_world_flat = cam_c2w @ points_cam_flat  # (4, H*W)
    points_world = points_world_flat.T.reshape(-1, 4)  # 转置并转为(H*W, 4)
    
    return points_world[:, :3]  # 返回XYZ世界坐标 (N, 3)

def set_ply_color(basename, radius=0.005):
    obj = bpy.data.objects[basename]
    bpy.context.view_layer.objects.active = obj

    geo_node_modifier = obj.modifiers.new(name='GeometryNodes', type='NODES')
    bpy.ops.node.new_geometry_node_group_assign()

    node_tree = geo_node_modifier.node_group
    group_input = node_tree.nodes['Group Input']
    group_output = node_tree.nodes['Group Output']

    mesh2points = node_tree.nodes.new(type='GeometryNodeMeshToPoints')
    mesh2points.inputs['Radius'].default_value = radius
    node_tree.links.new(group_input.outputs[0], mesh2points.inputs[0])

    setMaterial = node_tree.nodes.new(type='GeometryNodeSetMaterial')
    node_tree.links.new(mesh2points.outputs[0], setMaterial.inputs['Geometry'])
    node_tree.links.new(setMaterial.outputs['Geometry'], group_output.inputs[0])


    point_material = bpy.data.materials.new(name="PointMaterial")
    point_material.use_nodes = True
    # obj.data.materials[0] = point_material
    if obj.data.materials:
        obj.data.materials[0] = point_material
    else:
        obj.data.materials.append(point_material)

    setMaterial.inputs['Material'].default_value = point_material

    MatNodeTree = point_material.node_tree

    MatNodeTree.nodes.clear()

    attribute_node = MatNodeTree.nodes.new(type='ShaderNodeAttribute')
    attribute_node.attribute_name = 'Col'
    attribute_node.attribute_type = 'GEOMETRY'

    shader_node = MatNodeTree.nodes.new(type='ShaderNodeBsdfPrincipled')
    MatNodeTree.links.new(attribute_node.outputs['Color'], shader_node.inputs['Base Color'])

    shader_node.inputs['Metallic'].default_value = 0.77

    shader_output = MatNodeTree.nodes.new(type='ShaderNodeOutputMaterial')
    MatNodeTree.links.new(shader_node.outputs['BSDF'], shader_output.inputs['Surface'])

if __name__ == '__main__':
    parser = get_parser()
    parser.add_argument('--vis_all', action='store_true')
    args = parse_args(parser)

    setup()
    set_camera(location=(3, 0, 2.5), center=(0, 0, 1), focal=30)
    add_sunlight(name='Light', location=(0., 0., 5.), rotation=(0., np.pi/12, 0))

    data = np.load(args.path)

    # cam_c2w: (N, 4, 4)
    cam_c2w = data['cam_c2w']
    # 设置场景的开始和结束帧
    num_frames = cam_c2w.shape[0]
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = num_frames
    
    # 设置当前帧为第一帧
    bpy.context.scene.frame_current = 1
    # 设置相机
    camera_obj = bpy.data.objects['Camera']
    for i in range(cam_c2w.shape[0]):
        bpy.context.scene.frame_set(i + 1)
        cam_c2w_i = cam_c2w[i]
        cam_w2c_i = np.linalg.inv(cam_c2w_i)
        set_extrinsic(cam_w2c_i[:3, :3], cam_w2c_i[:3, 3], camera_obj)
        camera_obj.keyframe_insert('location', frame=i + 1)
        camera_obj.keyframe_insert('rotation_euler', frame=i + 1)
    
    # images: (N, H, W, 3)
    # depth: (N, H, W)
    # K: (3, 3)
    K = data['intrinsic']
    set_intrinsic(K, camera_obj, image_width=data['images'].shape[2], image_height=data['images'].shape[1])
    # 创造第一帧的点云
    basename = os.path.basename(args.path).split('.')[0]
    for i in range(data['images'].shape[0]):
        rgb = data['images'][i].reshape(-1, 3).astype(np.float32) / 255.
        depth = data['depths'][i]
        xyz = depth_to_xyz(depth, K, cam_c2w[i])
        pcd = trimesh.PointCloud(vertices=xyz, colors=rgb)
        pcd_name = f'/tmp/{basename}_{i}.ply'
        pcd.export(pcd_name)
        bpy.ops.wm.ply_import(filepath=pcd_name)
        
        # 获取导入的点云对象
        point_cloud_obj = bpy.context.selected_objects[0]
        # 重命名点云对象
        point_cloud_obj.name = f'{basename}_{i}'
        
        # 设置点云颜色
        set_ply_color(point_cloud_obj.name)
        
        # 设置点云只在当前帧可见
        # 默认在所有帧都不可见
        if not args.vis_all:
            point_cloud_obj.hide_render = True
            point_cloud_obj.hide_viewport = True
            point_cloud_obj.keyframe_insert('hide_render', frame=0)
            point_cloud_obj.keyframe_insert('hide_viewport', frame=0)
            
            # 在当前帧设置为可见
            current_frame = i + 1
            point_cloud_obj.hide_render = False
            point_cloud_obj.hide_viewport = False
            point_cloud_obj.keyframe_insert('hide_render', frame=current_frame)
            point_cloud_obj.keyframe_insert('hide_viewport', frame=current_frame)
            
            # 在下一帧设置为不可见
            point_cloud_obj.hide_render = True
            point_cloud_obj.hide_viewport = True
            point_cloud_obj.keyframe_insert('hide_render', frame=current_frame + 1)
            point_cloud_obj.keyframe_insert('hide_viewport', frame=current_frame + 1)

    # create_plane(vid=0, radius=2, center=(-3, 0))
    # create_points(vid=1, center=(0,-1, 1), alpha=0.5)
    # create_points(vid=2, center=(0, 1, 1), alpha=1)

    # # setup render
    # set_cycles_renderer(
    #     bpy.context.scene,
    #     bpy.data.objects["Camera"],
    #     num_samples=args.num_samples,
    #     use_transparent_bg=False,
    #     use_denoising=args.denoising,
    # )

    # n_parallel = 1
    # set_output_properties(bpy.context.scene, output_file_path=args.out, 
    #     res_x=args.res_x, res_y=args.res_y, 
    #     tile_x=args.res_x//n_parallel, tile_y=args.res_y, resolution_percentage=100,
    #     format='JPEG')
    # bpy.ops.render.render(write_still=True, animation=False)
    # if args.out_blend is not None:
    #     bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)