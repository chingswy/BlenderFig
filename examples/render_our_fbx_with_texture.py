import os
import bpy
import numpy as np
import shutil
from mathutils import Vector

from myblender.setup import (
    get_parser,
    parse_args,
    setup,
    add_sunlight,
    setLight_ambient,
    set_cycles_renderer,
    set_eevee_renderer,
    set_output_properties,
)
from myblender.geometry import (
    set_camera,
    build_plane,
    addGround,
)
from myblender.material import colorObj, setMat_plastic, setHDREnv
from myblender.camera import get_calibration_matrix_K_from_blender, get_3x4_RT_matrix_from_blender

def set_scene_frame_range(armature):
    if armature and armature.animation_data and armature.animation_data.action:
        # Get animation frame range from the action
        action = armature.animation_data.action
        frame_start = int(action.frame_range[0])
        frame_end = int(action.frame_range[1])

        # Set scene frame range to match the animation
        bpy.context.scene.frame_start = frame_start
        bpy.context.scene.frame_end = frame_end

        print(f"Animation frames set: {frame_start} to {frame_end}")
    else:
        print("No animation data found in the imported FBX")

def set_world_background():
    # Set scene background color to pure white
    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes
    world_links = bpy.context.scene.world.node_tree.links

    # Clear existing nodes
    for node in world_nodes:
        world_nodes.remove(node)

    # Create new nodes for pure white background
    background_node = world_nodes.new(type='ShaderNodeBackground')
    output_node = world_nodes.new(type='ShaderNodeOutputWorld')

    # Set background color to pure white (1,1,1)
    background_node.inputs['Color'].default_value = (1, 1, 1, 1)
    # Set strength for bright ambient light
    background_node.inputs['Strength'].default_value = 1.0

    # Connect nodes
    world_links.new(background_node.outputs['Background'], output_node.inputs['Surface'])

    # Position nodes in the node editor
    background_node.location = (-300, 0)
    output_node.location = (0, 0)

def calculate_mesh_center(mesh_object):
    vertices = mesh_object.data.vertices
    sum_co = Vector((0, 0, 0))
    for v in vertices:
        sum_co += v.co
    mesh_center = sum_co / len(vertices) if len(vertices) > 0 else Vector((0, 0, 0))
    return mesh_center

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

def find_center_of_mesh(mesh_object):
    world_bound_box = [mesh_object.matrix_world @ Vector(corner) for corner in mesh_object.bound_box]
    min_x, min_y, min_z = world_bound_box[0]
    max_x, max_y, max_z = world_bound_box[6]
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    return (center_x, center_y, center_z), min_z

def find_center_of_armature(armature):
    root_bone = armature.pose.bones[1] # 1 是pelvis
    world_matrix = armature.matrix_world @ root_bone.matrix
    position = world_matrix.translation
    return (position.x, position.y, position.z)

def set_texture_map(mesh_obj, body_texture='./assets/T_SM_SmplX_BaseColor.png'):
    """
    Set texture maps for the mesh object.

    Args:
        mesh_obj: Blender mesh object
        body_texture: Path to texture file or directory containing texture maps
    """
    material = bpy.data.materials.new(name="Custom_Texture")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    # Get the Principled BSDF node
    bsdf = nodes["Principled BSDF"]

    # Check if body_texture is a directory or a single file
    if os.path.isdir(body_texture):
        # Directory mode - look for diffuse, roughness, and normal maps
        diffuse_path = None
        roughness_path = None
        normal_path = None

        # Search for texture files in the directory
        for filename in os.listdir(body_texture):
            file_path = os.path.join(body_texture, filename)
            if not os.path.isfile(file_path):
                continue

            filename_lower = filename.lower()
            if 'diffuse' in filename_lower or 'basecolor' in filename_lower or 'albedo' in filename_lower:
                diffuse_path = file_path
            elif 'roughness' in filename_lower:
                roughness_path = file_path
            elif 'normal' in filename_lower:
                normal_path = file_path

        # Load and connect diffuse map
        if diffuse_path:
            print(f"Loading diffuse map: {diffuse_path}")
            diffuse_image = bpy.data.images.load(diffuse_path)
            diffuse_node = nodes.new("ShaderNodeTexImage")
            diffuse_node.image = diffuse_image
            diffuse_node.location = (-400, 300)
            links.new(diffuse_node.outputs["Color"], bsdf.inputs["Base Color"])

        # Load and connect roughness map
        if roughness_path:
            print(f"Loading roughness map: {roughness_path}")
            roughness_image = bpy.data.images.load(roughness_path)
            roughness_node = nodes.new("ShaderNodeTexImage")
            roughness_node.image = roughness_image
            roughness_node.location = (-400, 0)
            # Set color space to Non-Color for roughness map
            roughness_node.image.colorspace_settings.name = 'Non-Color'
            links.new(roughness_node.outputs["Color"], bsdf.inputs["Roughness"])

        # Load and connect normal map
        if normal_path:
            print(f"Loading normal map: {normal_path}")
            normal_image = bpy.data.images.load(normal_path)
            normal_tex_node = nodes.new("ShaderNodeTexImage")
            normal_tex_node.image = normal_image
            normal_tex_node.location = (-400, -300)
            # Set color space to Non-Color for normal map
            normal_tex_node.image.colorspace_settings.name = 'Non-Color'

            # Create normal map node
            normal_map_node = nodes.new("ShaderNodeNormalMap")
            normal_map_node.location = (-200, -300)

            # Connect normal texture to normal map node, then to BSDF
            links.new(normal_tex_node.outputs["Color"], normal_map_node.inputs["Color"])
            links.new(normal_map_node.outputs["Normal"], bsdf.inputs["Normal"])

    elif os.path.isfile(body_texture):
        # Single file mode - only load diffuse texture
        print(f"Loading single texture: {body_texture}")
        texture_image = bpy.data.images.load(body_texture)
        tex_image = nodes.new("ShaderNodeTexImage")
        tex_image.image = texture_image
        tex_image.location = (-400, 300)
        links.new(tex_image.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        print(f"Warning: Texture path not found: {body_texture}")
        # Create a simple material with default color
        bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)

    # Apply material to the mesh object
    if mesh_obj.data.materials:
        mesh_obj.data.materials[0] = material
    else:
        mesh_obj.data.materials.append(material)

    return material

def get_KRT_from_camera(camera):
    camera_list = []
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1):
        bpy.context.scene.frame_set(frame)
        K = get_calibration_matrix_K_from_blender(camera, mode='complete')
        RT = get_3x4_RT_matrix_from_blender(camera)
        camera_list.append({
            'K': K,
            'RT': RT,
        })
    return camera_list

def get_temp_output_path(final_output_path, temp_dir):
    """
    Get temporary output path based on final output path and temp directory.
    If temp_dir is None, return the original path.
    """
    if temp_dir is None:
        return final_output_path

    # Ensure temp directory exists
    os.makedirs(temp_dir, exist_ok=True)

    # Get the basename of the final output
    basename = os.path.basename(final_output_path)
    temp_path = os.path.join(temp_dir, basename)

    return temp_path

def copy_from_temp_to_final(temp_path, final_path):
    """
    Copy file from temporary location to final location.
    Create final directory if it doesn't exist.
    """
    if temp_path == final_path:
        # No temporary directory was used
        return

    if not os.path.exists(temp_path):
        print(f"Warning: Temporary file {temp_path} does not exist")
        return

    # Ensure final directory exists
    final_dir = os.path.dirname(final_path)
    os.makedirs(final_dir, exist_ok=True)

    # Copy file
    shutil.copy2(temp_path, final_path)
    print(f"Copied {temp_path} -> {final_path}")

    # Optionally remove temp file
    try:
        os.remove(temp_path)
        print(f"Removed temporary file {temp_path}")
    except OSError as e:
        print(f"Warning: Could not remove temporary file {temp_path}: {e}")


def get_center_of_all(armature, mesh_object, look_at_mode='root'):
    min_height = 10
    # 检查一遍整段的位置
    center_list = []
    frame_list = list(range(bpy.context.scene.frame_start, bpy.context.scene.frame_end, 5)) + [bpy.context.scene.frame_end]
    for frame in frame_list:
        bpy.context.scene.frame_set(frame)
        if look_at_mode == 'root':
            center = find_center_of_armature(armature)
        elif look_at_mode == 'mesh':
            center, _ = find_center_of_mesh(mesh_object)
        _, min_z = find_center_of_mesh(mesh_object)
        min_height = min(min_height, min_z)
        center_list.append(center)
    center_list = np.array(center_list)
    print(f'>>> Frame list: {frame_list}')
    print(f'>>> Center list: {center_list.min(axis=0)}, {center_list.max(axis=0)}')
    print(f'>>> Min height: {min_height}')
    return frame_list, center_list, min_height

def get_camera_center_of_all(center_list, max_xy_movement=2.0, max_z_movement=1.0, view_angle=(1, 0, 0)):
    """
    根据一系列3D中心点的运动轨迹，计算相机的中心位置。
    如果运动范围较小，则返回一个静态的全局相机位置；否则返回一个动态跟随的相机位置。
    """
    center_list = np.array(center_list)

    # 1. 判断Z轴是否应为静态
    min_height, max_height = center_list[:, 2].min(), center_list[:, 2].max()
    is_static_z = (max_height - min_height) < max_z_movement

    # 2. 判断XY平面是否应为静态
    min_xy, max_xy = center_list[:, :2].min(axis=0), center_list[:, :2].max(axis=0)
    movement_span_xy = np.linalg.norm(max_xy - min_xy)
    is_static_xy = movement_span_xy < max_xy_movement

    print(f'>>> Z-axis movement: {max_height - min_height:.2f}, Is Static: {is_static_z}')
    print(f'>>> XY-plane movement span: {movement_span_xy:.2f}, Is Static: {is_static_xy}')

    # 3. 计算XY坐标列表
    if is_static_xy:
        # --- 计算一个固定的XY相机位置 ---
        view_angle_xy = np.array([view_angle[0], view_angle[1]]).astype(np.float32)
        norm = np.linalg.norm(view_angle_xy)
        if norm > 0:
            view_angle_xy /= norm
        else:
            view_angle_xy = np.array([1.0, 0.0])  # 默认看向X轴正方向

        perp_angle = np.array([-view_angle_xy[1], view_angle_xy[0]])

        # 计算运动轨迹的2D中心点
        center_point_xy = (min_xy + max_xy) / 2

        # 将所有点投影到以相机视角为基准的新坐标系
        points_relative = center_list[:, :2] - center_point_xy
        proj_along = np.dot(points_relative, view_angle_xy)
        proj_perp = np.dot(points_relative, perp_angle)

        range_along = proj_along.max() - proj_along.min()
        range_perp = proj_perp.max() - proj_perp.min()

        # NOTE: 这里的相机距离是一个经验公式，可能需要根据具体FOV调整
        # 它基于动作的最大范围（宽度或深度），并额外后退以确保最前方的点也在视野内。
        camera_distance = max(range_along, range_perp) #+ abs(proj_along.max())

        camera_xy = center_point_xy - camera_distance * view_angle_xy

        # 为所有帧设置相同的相机XY位置
        xy_list = np.tile(camera_xy, (len(center_list), 1))
    else:
        # 动态跟随模式
        xy_list = center_list[:, :2]

    # 4. 计算Z坐标列表
    if is_static_z:
        # 固定在最高点
        z_list = np.full(len(center_list), max_height)
    else:
        # 动态跟随模式
        z_list = center_list[:, 2]

    # 5. 组合最终结果
    return np.hstack([xy_list, z_list.reshape(-1, 1)])

if __name__ == '__main__':
    # ${blender} --background -noaudio --python examples/render_our_fbx.py -- ~/Desktop/t2m/swimanimset_jog_fwd_in_shallow_water.fbx
    parser = get_parser()
    parser.add_argument('--hdr', type=str, default=None)
    parser.add_argument('--camera_height', type=float, default=0.2)
    parser.add_argument('--add_sideview', action='store_true')
    parser.add_argument('--add_topview', action='store_true')
    parser.add_argument('--blur', action='store_true')
    parser.add_argument('--static', action='store_true')
    parser.add_argument('--eevee', action='store_true')
    parser.add_argument('--checkerboard', action='store_true')
    args = parse_args(parser)

    setup()

    # set_world_background()

    # setHDREnv(fn='../DCC_Scripts/blender/Zbyg-Studio_0018_1k_m.hdr', strength=1.0)
    if args.hdr:
        setHDREnv(fn=args.hdr, strength=1.0)
    else:
        add_sunlight(name='Light', location=(0., 0., 5.), rotation=(0., np.pi/12, 0), strength=2.0)

    setLight_ambient(color=(0.3,0.3,0.3,1))

    fbx_path = args.path
    assert os.path.exists(fbx_path), fbx_path
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    # Get the imported objects
    obj_names = [o.name for o in bpy.context.selected_objects]

    armature, mesh_object, mesh_object_list = find_armature_and_mesh(obj_names)

    set_scene_frame_range(armature)

    look_at_mode = 'root' # 'root' or 'mesh'
    # base_location = (3, 0, 0.5)
    base_distance = 2.
    view_angle_front = (0, 1, 0)
    view_angle_side = (1, 0, 0)
    height_offset = -0.1
    camera_height = 0.2
    # set_camera(location=(0, -4, 2.), center=(0, 0, 1), focal=30)
    side_camera = bpy.data.cameras.new(name="SideCamera")
    side_camera_obj = bpy.data.objects.new(name="SideCamera", object_data=side_camera)
    bpy.context.collection.objects.link(side_camera_obj)

    top_camera = bpy.data.cameras.new(name="TopCamera")
    top_camera_obj = bpy.data.objects.new(name="TopCamera", object_data=top_camera)
    bpy.context.collection.objects.link(top_camera_obj)

    frame_list, center_list, min_height = get_center_of_all(armature, mesh_object, look_at_mode)
    camera_center_list = get_camera_center_of_all(center_list, max_xy_movement=5.0, max_z_movement=1.0, view_angle=view_angle_front)
    side_camera_center_list = get_camera_center_of_all(center_list, max_xy_movement=5.0, max_z_movement=1.0, view_angle=view_angle_side)
    dist = np.linalg.norm(center_list[:, None, :] - center_list[None, :, :], axis=-1).max()


    for index, frame in enumerate(frame_list):
        # Set the current frame to the last frame
        bpy.context.scene.frame_set(frame)
        # Update camera to look at the last frame position
        center = camera_center_list[index]
        set_camera(
            location=(center[0] - base_distance * view_angle_front[0], center[1] - base_distance * view_angle_front[1], center[2] + camera_height + height_offset),
            center=(center[0], center[1], center[2] + height_offset), focal=30, frame=frame,
            camera=bpy.data.objects["Camera"],
        )
        if args.add_sideview:
            center_side = side_camera_center_list[index]
            set_camera(
                location=(center_side[0] - base_distance * view_angle_side[0], center_side[1] - base_distance * view_angle_side[1], center_side[2] + camera_height + height_offset),
                center=(center_side[0], center_side[1], center_side[2] + height_offset), focal=30, frame=frame,
                camera=bpy.data.objects["SideCamera"],
            )
        if args.add_topview:
            set_camera(
                location=(base_location_top[0] + center[0], base_location_top[1] + center[1], base_location_top[2] + center[2]),
                center=(center[0], center[1], center[2]), focal=30, frame=frame,
                camera=bpy.data.objects["TopCamera"],
            )
    # Smooth camera motion using spline interpolation
    camera_obj = bpy.data.objects["Camera"]

    # Select the camera object
    bpy.ops.object.select_all(action='DESELECT')
    camera_obj.select_set(True)
    bpy.context.view_layer.objects.active = camera_obj

    # Get the animation data
    if camera_obj.animation_data and camera_obj.animation_data.action:
        # Apply spline interpolation to the camera animation curves
        for fcurve in camera_obj.animation_data.action.fcurves:
            for kf in fcurve.keyframe_points:
                kf.interpolation = 'BEZIER'

        # If side camera exists, also smooth its motion
        if args.add_sideview:
            if side_camera_obj.animation_data and side_camera_obj.animation_data.action:
                for fcurve in side_camera_obj.animation_data.action.fcurves:
                    for kf in fcurve.keyframe_points:
                        kf.interpolation = 'BEZIER'

        # If top camera exists, also smooth its motion
        if args.add_topview:
            if top_camera_obj.animation_data and top_camera_obj.animation_data.action:
                for fcurve in top_camera_obj.animation_data.action.fcurves:
                    for kf in fcurve.keyframe_points:
                        kf.interpolation = 'BEZIER'

    # 当有HDR时，地面作为阴影捕捉器（不可见但接收阴影）；无HDR时地面正常显示
    shadow_catcher_mode = args.hdr is not None
    plane_size = max(int(dist * 2), 10)
    if args.hdr:
        bpy.ops.mesh.primitive_plane_add(size=20, enter_editmode=False, align='WORLD', location=(cx, cy, min_height), scale=(1, 1, 1))
        ground_mesh = bpy.context.object
        ground_mesh.name = "ground"
        ground_mesh.is_shadow_catcher = True
    elif args.checkerboard:
        build_plane(translation=(center_list[0, 0], center_list[0, 1], min_height), plane_size=plane_size)
    else:
        ground_mesh = addGround(
            location=(center_list[0, 0], center_list[0, 1], min_height),
            groundSize=plane_size,
            shadowBrightness=0.1,
            normal_axis="z",
            alpha=1,
            tex_fn=os.path.join('assets', 'cyclesProceduralWoodFloor.png'),
            shadow_catcher=shadow_catcher_mode,
        )

    # Apply material to the mesh object, not the armature
    if mesh_object:
        for mesh_obj_ in mesh_object_list:
            if False:
                meshColor = colorObj((153/255.,  216/255.,  201/255., 1.), 0.5, 1.0, 1.0, 0.0, 2.0)
                setMat_plastic(mesh_obj_, meshColor, roughness=0.9, metallic=0.5, specular=0.5)
            else:
                # set_texture_map(mesh_obj_, body_texture='assets/SMPLitex-texture-00000.png')
                set_texture_map(mesh_obj_, body_texture='assets/T_SM_SmplX_BaseColor.png')
                # set_texture_map(mesh_obj_, body_texture='assets/smplx_texture_f_alb.png')
                # set_texture_map(mesh_obj_, body_texture='/Users/shuaiqing/Downloads/bedlam_body_textures_meshcapade/smpl/MC_texture_skintones/male/skin/skin_m_asian_01_ALB.png')
                # set_texture_map(mesh_obj_, body_texture='/Users/shuaiqing/Downloads/bedlam_body_textures_meshcapade/smpl/MC_texture_skintones/female/skin/skin_f_asian_01_ALB.png')
                # set_texture_map(mesh_obj_, body_texture='assets/meshcapade_texture')
                # set_texture_map(mesh_obj_, body_texture='assets/meshtextures/mc-female-casual')

    # First render with the main camera
    if args.eevee:
        set_eevee_renderer(bpy.context.scene, bpy.data.objects["Camera"])
    else:
        set_cycles_renderer(
            bpy.context.scene,
            bpy.data.objects["Camera"],
            num_samples=args.num_samples,
            use_transparent_bg=False,
            use_denoising=args.denoising,
            use_adaptive_sampling=True,
            use_motion_blur=args.blur,
        )

    # Get temporary output path for main video
    temp_output_path = get_temp_output_path(args.out, args.tmp)
    set_output_properties(bpy.context.scene, output_file_path=temp_output_path,
        res_x=args.res_x, res_y=args.res_y,
        tile_x=args.res_x, tile_y=args.res_y,
        resolution_percentage=100,
        format='FFMPEG',
    )

    # Save camera parameters for each frame
    # Get the active camera
    camera_list = get_KRT_from_camera(bpy.data.objects["Camera"])
    K = np.stack([camera['K'] for camera in camera_list], axis=0)
    RT = np.stack([camera['RT'] for camera in camera_list], axis=0)
    camera_npz_path = args.out.replace('.mp4', '_camera.npz')
    temp_camera_npz_path = get_temp_output_path(camera_npz_path, args.tmp)
    np.savez(temp_camera_npz_path, K=K, RT=RT)

    if not args.debug:
        bpy.ops.render.render(animation=True)
        # Copy main video and camera file from temp to final location
        copy_from_temp_to_final(temp_output_path, args.out)
        copy_from_temp_to_final(temp_camera_npz_path, camera_npz_path)

    if args.add_sideview:
        sideview_name = os.path.join(
            os.path.dirname(args.out),
            os.path.basename(args.out).split('.')[0] + '-side.mp4'
        )
        sideview_camera_list = get_KRT_from_camera(bpy.data.objects["SideCamera"])
        K = np.stack([camera['K'] for camera in sideview_camera_list], axis=0)
        RT = np.stack([camera['RT'] for camera in sideview_camera_list], axis=0)
        sideview_camera_npz_path = sideview_name.replace('.mp4', '_camera.npz')
        temp_sideview_camera_npz_path = get_temp_output_path(sideview_camera_npz_path, args.tmp)
        np.savez(temp_sideview_camera_npz_path, K=K, RT=RT)

        if args.eevee:
            set_eevee_renderer(bpy.context.scene, bpy.data.objects["SideCamera"])
        else:
            set_cycles_renderer(
                bpy.context.scene,
                bpy.data.objects["SideCamera"],
                num_samples=args.num_samples,
                use_transparent_bg=False,
                use_denoising=args.denoising,
                use_adaptive_sampling=True,
                use_motion_blur=args.blur,
            )

        # Get temporary output path for side view video
        temp_sideview_path = get_temp_output_path(sideview_name, args.tmp)
        set_output_properties(bpy.context.scene, output_file_path=temp_sideview_path,
            res_x=args.res_x, res_y=args.res_y,
            tile_x=args.res_x, tile_y=args.res_y,
            resolution_percentage=100,
            format='FFMPEG',
        )
        if not args.debug:
            bpy.ops.render.render(animation=True)
            # Copy side view video and camera file from temp to final location
            copy_from_temp_to_final(temp_sideview_path, sideview_name)
            copy_from_temp_to_final(temp_sideview_camera_npz_path, sideview_camera_npz_path)

    if args.add_topview:
        topview_name = os.path.join(
            os.path.dirname(args.out),
            os.path.basename(args.out).split('.')[0] + '-top.mp4'
        )
        topview_camera_list = get_KRT_from_camera(bpy.data.objects["TopCamera"])
        K = np.stack([camera['K'] for camera in topview_camera_list], axis=0)
        RT = np.stack([camera['RT'] for camera in topview_camera_list], axis=0)
        topview_camera_npz_path = topview_name.replace('.mp4', '_camera.npz')
        temp_topview_camera_npz_path = get_temp_output_path(topview_camera_npz_path, args.tmp)
        np.savez(temp_topview_camera_npz_path, K=K, RT=RT)

        set_cycles_renderer(
            bpy.context.scene,
            bpy.data.objects["TopCamera"],
            num_samples=args.num_samples,
            use_transparent_bg=False,
            use_denoising=args.denoising,
            use_adaptive_sampling=True,
            use_motion_blur=args.blur,
        )

        # Get temporary output path for top view video
        temp_topview_path = get_temp_output_path(topview_name, args.tmp)
        set_output_properties(bpy.context.scene, output_file_path=temp_topview_path,
            res_x=args.res_x, res_y=args.res_y,
            tile_x=args.res_x, tile_y=args.res_y,
            resolution_percentage=100,
            format='FFMPEG',
        )
        if not args.debug:
            bpy.ops.render.render(animation=True)
            # Copy top view video and camera file from temp to final location
            copy_from_temp_to_final(temp_topview_path, topview_name)
            copy_from_temp_to_final(temp_topview_camera_npz_path, topview_camera_npz_path)
