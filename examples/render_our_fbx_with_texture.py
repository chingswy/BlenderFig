import os
import bpy
import numpy as np
from mathutils import Vector

from myblender.setup import (
    get_parser,
    parse_args,
    setup,
    add_sunlight,
    setLight_ambient,
    set_cycles_renderer,
    set_output_properties,
)
from myblender.geometry import (
    set_camera,
    build_plane,
    addGround,
)
from myblender.material import colorObj, setMat_plastic, setHDREnv


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

def set_eevee_renderer():

    # Set up Eevee renderer
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'

    # Configure Eevee settings for faster rendering
    scene.eevee.taa_render_samples = 16  # Reduce samples for faster rendering
    scene.eevee.use_soft_shadows = True
    scene.eevee.use_ssr = False  # Disable screen space reflections for speed
    scene.eevee.use_ssr_refraction = False
    scene.eevee.use_gtao = True  # Keep ambient occlusion for better visuals
    scene.eevee.gtao_distance = 0.2
    scene.eevee.use_bloom = False  # Disable bloom for speed

    # Set up camera for rendering
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 100

    return scene


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

def set_texture_map(mesh_obj, body_texture='./assets/T_SM_SmplX_BaseColor.png'):            # 创建材质
    if True:
        material = bpy.data.materials.new(name="Custom_Texture")
        material.use_nodes = True
        bsdf = material.node_tree.nodes["Principled BSDF"]

        # 加载纹理
        texture_image = bpy.data.images.load(body_texture)  # 纹理文件路径
        tex_image = material.node_tree.nodes.new("ShaderNodeTexImage")
        tex_image.image = texture_image

        # 连接纹理到材质
        material.node_tree.links.new(bsdf.inputs["Base Color"], tex_image.outputs["Color"])

        # 应用材质到模型
        if mesh_obj.data.materials:
            mesh_obj.data.materials[0] = material
        else:
            mesh_obj.data.materials.append(material)
    return material

if __name__ == '__main__':
    # ${blender} --background -noaudio --python examples/render_our_fbx.py -- ~/Desktop/t2m/swimanimset_jog_fwd_in_shallow_water.fbx
    parser = get_parser()
    parser.add_argument('--add_sideview', action='store_true')
    parser.add_argument('--add_topview', action='store_true')
    parser.add_argument('--blur', action='store_true')
    parser.add_argument('--static', action='store_true')
    args = parse_args(parser)

    setup()
    add_sunlight(name='Light', location=(0., 0., 5.), rotation=(0., np.pi/12, 0), strength=2.0)

    # set_world_background()

    # setHDREnv(fn='../DCC_Scripts/blender/Zbyg-Studio_0018_1k_m.hdr', strength=1.0)
    setLight_ambient(color=(0.6,0.6,0.6,1))
    # scene = set_eevee_renderer()

    fbx_path = args.path
    assert os.path.exists(fbx_path), fbx_path
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    # Get the imported objects
    obj_names = [o.name for o in bpy.context.selected_objects]

    armature, mesh_object, mesh_object_list = find_armature_and_mesh(obj_names)

    set_scene_frame_range(armature)

    look_at_mode = 'root' # 'root' or 'mesh'
    # base_location = (3, 0, 0.5)
    if look_at_mode == 'root':
        base_distance = 2.5
        base_location = (0, -base_distance, 0.5)
        base_location_side = (-base_distance, 0, 0.5)
        base_location_top = (0, 0, base_distance + 1.0)  # 俯视图相机位置：在目标正上方
    elif look_at_mode == 'mesh':
        base_distance = 2.5
        base_location = (0, -base_distance, 0.5)
        base_location_side = (-base_distance, 0, 0.5)
        base_location_top = (0, 0, base_distance + 1.0)  # 俯视图相机位置：在目标正上方

    # set_camera(location=(0, -4, 2.), center=(0, 0, 1), focal=30)
    _, min_z = find_center_of_mesh(mesh_object)
    center = find_center_of_armature(armature)
    cx, cy = center[0], center[1]

    side_camera = bpy.data.cameras.new(name="SideCamera")
    side_camera_obj = bpy.data.objects.new(name="SideCamera", object_data=side_camera)
    bpy.context.collection.objects.link(side_camera_obj)

    top_camera = bpy.data.cameras.new(name="TopCamera")
    top_camera_obj = bpy.data.objects.new(name="TopCamera", object_data=top_camera)
    bpy.context.collection.objects.link(top_camera_obj)

    min_height = 0
    # 检查一遍整段的位置
    center_list = []
    for frame in list(range(bpy.context.scene.frame_start, bpy.context.scene.frame_end, 30)) + [bpy.context.scene.frame_end]:
        bpy.context.scene.frame_set(frame)
        if look_at_mode == 'root':
            center = find_center_of_armature(armature)
        elif look_at_mode == 'mesh':
            center, _ = find_center_of_mesh(mesh_object)
        _, min_z = find_center_of_mesh(mesh_object)
        min_height = min(min_height, min_z)
        center_list.append(center)
    center_list = np.array(center_list)
    dist = np.linalg.norm(center_list[1:] - center_list[:1], axis=-1).max()
    flag_use_static_camera = args.static
    if dist < 0.5:
        flag_use_static_camera = True

    for frame in list(range(bpy.context.scene.frame_start, bpy.context.scene.frame_end, 30)) + [bpy.context.scene.frame_end]:
        # Set the current frame to the last frame
        bpy.context.scene.frame_set(frame)

        if look_at_mode == 'root':
            center = find_center_of_armature(armature)
        elif look_at_mode == 'mesh':
            center, _ = find_center_of_mesh(mesh_object)
        _, min_z = find_center_of_mesh(mesh_object)
        min_height = min(min_height, min_z)
        # Update camera to look at the last frame position
        if flag_use_static_camera and frame > bpy.context.scene.frame_start:
            # skip setting camera
            continue
        set_camera(
            location=(base_location[0] + center[0], base_location[1] + center[1], base_location[2] + center[2]),
            center=(center[0], center[1], center[2]), focal=30, frame=frame,
            camera=bpy.data.objects["Camera"],
        )
        if args.add_sideview:
            set_camera(
                location=(base_location_side[0] + center[0], base_location_side[1] + center[1], base_location_side[2] + center[2]),
                center=(center[0], center[1], center[2]), focal=30, frame=frame,
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

    # build_plane(translation=(0, 0, 0), plane_size=20)
    ground_mesh = addGround(
        location=(cx, cy, min_height),
        groundSize=20,
        shadowBrightness=0.1,
        normal_axis="z",
        alpha=1,
        tex_fn=os.path.join('assets', 'cyclesProceduralWoodFloor.png'),
    )

    # Apply material to the mesh object, not the armature
    if mesh_object:
        for mesh_obj_ in mesh_object_list:
            if False:
                meshColor = colorObj((153/255.,  216/255.,  201/255., 1.), 0.5, 1.0, 1.0, 0.0, 2.0)
                setMat_plastic(mesh_obj_, meshColor, roughness=0.9, metallic=0.5, specular=0.5)
            else:
                # set_texture_map(mesh_obj_, body_texture='assets/SMPLitex-texture-00000.png')
                # set_texture_map(mesh_obj_, body_texture='assets/T_SM_SmplX_BaseColor.png')
                set_texture_map(mesh_obj_, body_texture='assets/smplx_texture_f_alb.png')

    # First render with the main camera
    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=args.num_samples,
        use_transparent_bg=False,
        use_denoising=False,
        use_adaptive_sampling=True,
        use_motion_blur=args.blur,
    )
    set_output_properties(bpy.context.scene, output_file_path=args.out,
        res_x=args.res_x, res_y=args.res_y,
        tile_x=args.res_x, tile_y=args.res_y,
        resolution_percentage=100,
        format='FFMPEG',
    )
    if not args.debug:
        bpy.ops.render.render(animation=True)

    if args.add_sideview:
        sideview_name = os.path.join(
            os.path.dirname(args.out),
            os.path.basename(args.out).split('.')[0] + '-side.mp4'
        )
        set_cycles_renderer(
            bpy.context.scene,
            bpy.data.objects["SideCamera"],
            num_samples=args.num_samples,
            use_transparent_bg=False,
            use_denoising=False,
            use_adaptive_sampling=True,
            use_motion_blur=args.blur,
        )

        set_output_properties(bpy.context.scene, output_file_path=sideview_name,
            res_x=args.res_x, res_y=args.res_y,
            tile_x=args.res_x, tile_y=args.res_y,
            resolution_percentage=100,
            format='FFMPEG',
        )
        if not args.debug:
            bpy.ops.render.render(animation=True)

    if args.add_topview:
        topview_name = os.path.join(
            os.path.dirname(args.out),
            os.path.basename(args.out).split('.')[0] + '-top.mp4'
        )
        set_cycles_renderer(
            bpy.context.scene,
            bpy.data.objects["TopCamera"],
            num_samples=args.num_samples,
            use_transparent_bg=False,
            use_denoising=False,
            use_adaptive_sampling=True,
            use_motion_blur=args.blur,
        )

        set_output_properties(bpy.context.scene, output_file_path=topview_name,
            res_x=args.res_x, res_y=args.res_y,
            tile_x=args.res_x, tile_y=args.res_y,
            resolution_percentage=100,
            format='FFMPEG',
        )
        if not args.debug:
            bpy.ops.render.render(animation=True)
