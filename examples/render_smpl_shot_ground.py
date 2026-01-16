'''
  @ Date: 2024-12-24
  @ Author: Qing Shuai
  @ Description: Render a sequence of sampled frames from an animation file (FBX)
                 arranged from left to right along the x-axis as a single shot.
                 Optimized for SIGGRAPH paper figure quality.
'''
import os
import numpy as np
import bpy
from myblender.geometry import (
    set_camera,
    build_plane,
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

# ============================================================================
# SIGGRAPH-Quality Color Palettes (更饱和、更有质感的配色)
# ============================================================================
SIGGRAPH_PALETTES = {
    # Classic academic blue-gray palette (most common in SIGGRAPH) - 加深饱和度
    'academic': [
        (0.220, 0.380, 0.580, 1.0),  # Deep steel blue
        (0.780, 0.380, 0.220, 1.0),  # Rich terracotta
        (0.280, 0.520, 0.450, 1.0),  # Deep sage green
        (0.620, 0.350, 0.450, 1.0),  # Deep dusty rose
        (0.350, 0.300, 0.480, 1.0),  # Rich purple
        (0.680, 0.550, 0.350, 1.0),  # Warm sand
    ],
    # Clean minimal palette - 更深的色调
    'minimal': [
        (0.180, 0.380, 0.550, 1.0),  # Deep ocean blue
        (0.780, 0.420, 0.300, 1.0),  # Rich coral
        (0.250, 0.480, 0.420, 1.0),  # Deep teal
        (0.520, 0.350, 0.480, 1.0),  # Rich mauve
        (0.380, 0.380, 0.400, 1.0),  # Medium gray
        (0.620, 0.520, 0.380, 1.0),  # Warm tan
    ],
    # High contrast for clarity - 高饱和度
    'contrast': [
        (0.150, 0.320, 0.580, 1.0),  # Vivid deep blue
        (0.820, 0.350, 0.200, 1.0),  # Vivid burnt orange
        (0.180, 0.480, 0.350, 1.0),  # Vivid forest green
        (0.650, 0.220, 0.320, 1.0),  # Vivid berry
        (0.380, 0.280, 0.450, 1.0),  # Vivid plum
        (0.500, 0.480, 0.350, 1.0),  # Rich olive
    ],
    # Warm clay/sculpture look - 更深的粘土色
    'clay': [
        (0.580, 0.420, 0.350, 1.0),  # Rich clay base
        (0.480, 0.350, 0.280, 1.0),  # Deep clay
        (0.650, 0.500, 0.420, 1.0),  # Medium clay
        (0.520, 0.400, 0.320, 1.0),  # Warm mid clay
        (0.600, 0.450, 0.380, 1.0),  # Terracotta clay
        (0.550, 0.420, 0.350, 1.0),  # Natural clay
    ],
    # Cool professional - 专业冷色调
    'professional': [
        (0.220, 0.350, 0.520, 1.0),  # Deep slate blue
        (0.650, 0.380, 0.280, 1.0),  # Rich muted orange
        (0.300, 0.480, 0.420, 1.0),  # Deep seafoam
        (0.480, 0.320, 0.420, 1.0),  # Rich violet
        (0.380, 0.420, 0.480, 1.0),  # Cool slate
        (0.520, 0.480, 0.380, 1.0),  # Warm khaki
    ],
}


def set_siggraph_material(obj, color_rgba, use_sss=True, sss_weight=0.05):
    """
    Create a SIGGRAPH-quality material with rich texture and depth.
    Features ambient occlusion, subtle SSS, and professional shading.

    Args:
        obj: The Blender mesh object
        color_rgba: Base color as (R, G, B, A) tuple
        use_sss: Whether to use subsurface scattering
        sss_weight: Subsurface scattering weight (0.0-1.0)
    """
    mat_name = f"SIGGRAPH_Mat_{obj.name}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Create output node
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (600, 0)

    # Create Principled BSDF
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (200, 0)

    # ========== 添加环境光遮蔽 (AO) 增加层次感 ==========
    ao_node = nodes.new(type='ShaderNodeAmbientOcclusion')
    ao_node.location = (-400, 100)
    ao_node.inputs['Distance'].default_value = 0.5  # AO影响距离

    # Gamma节点调整AO强度
    gamma_node = nodes.new(type='ShaderNodeGamma')
    gamma_node.location = (-200, -50)
    gamma_node.inputs['Gamma'].default_value = 1.5  # AO强度

    # Mix RGB节点混合颜色和AO
    mix_node = nodes.new(type='ShaderNodeMixRGB')
    mix_node.location = (0, 100)
    mix_node.blend_type = 'MULTIPLY'
    mix_node.inputs['Fac'].default_value = 0.35  # AO混合强度

    # ========== 色彩调整 - 增加饱和度和对比度 ==========
    hsv_node = nodes.new(type='ShaderNodeHueSaturation')
    hsv_node.location = (-600, 100)
    hsv_node.inputs['Hue'].default_value = 0.5
    hsv_node.inputs['Saturation'].default_value = 1.20  # 增加饱和度
    hsv_node.inputs['Value'].default_value = 1.0  # 保持原始亮度
    hsv_node.inputs['Color'].default_value = color_rgba

    # 亮度/对比度调整
    bc_node = nodes.new(type='ShaderNodeBrightContrast')
    bc_node.location = (-400, -50)
    bc_node.inputs['Bright'].default_value = 0.0  # 保持亮度
    bc_node.inputs['Contrast'].default_value = 0.12  # 适中对比度

    # 连接颜色处理链
    links.new(hsv_node.outputs['Color'], bc_node.inputs['Color'])
    links.new(bc_node.outputs['Color'], ao_node.inputs['Color'])
    links.new(ao_node.outputs['Color'], mix_node.inputs['Color1'])
    links.new(ao_node.outputs['AO'], gamma_node.inputs['Color'])
    links.new(gamma_node.outputs['Color'], mix_node.inputs['Color2'])

    # 连接到Principled BSDF
    links.new(mix_node.outputs['Color'], principled.inputs['Base Color'])

    # ========== SIGGRAPH专业材质设置 ==========
    # 半光泽外观 - 不完全哑光，有一定质感
    principled.inputs['Roughness'].default_value = 0.42  # 适中的粗糙度，有光泽但不刺眼
    principled.inputs['Metallic'].default_value = 0.0    # 非金属
    principled.inputs['Specular'].default_value = 0.45   # 适中的高光
    principled.inputs['Sheen'].default_value = 0.15      # 轻微的绒面效果

    # 微表面细节 - Clearcoat增加层次感
    principled.inputs['Clearcoat'].default_value = 0.08  # 轻微的透明涂层
    principled.inputs['Clearcoat Roughness'].default_value = 0.3

    # Subsurface scattering - 减弱以保持颜色深度
    if use_sss:
        principled.inputs['Subsurface'].default_value = sss_weight
        # SSS颜色与基础色相近，避免变白
        sss_color = (
            color_rgba[0] * 0.95,
            color_rgba[1] * 0.85,
            color_rgba[2] * 0.80,
            1.0
        )
        principled.inputs['Subsurface Color'].default_value = sss_color
        principled.inputs['Subsurface Radius'].default_value = (0.5, 0.25, 0.15)

    # IOR for realistic fresnel
    principled.inputs['IOR'].default_value = 1.45

    # Link to output
    links.new(principled.outputs['BSDF'], output_node.inputs['Surface'])

    # Assign material to object
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

    return mat


def setup_siggraph_lighting(num_samples_width, key_strength=4.0, fill_ratio=0.3, rim_ratio=0.5):
    """
    Set up professional three-point lighting for SIGGRAPH-quality renders.
    优化版本：更亮的整体照明，更明显的阴影，避免过曝。

    Args:
        num_samples_width: Width of the scene to properly position lights
        key_strength: Main light strength
        fill_ratio: Fill light strength ratio relative to key
        rim_ratio: Rim light strength ratio relative to key
    """
    # ========== Key Light - 主光源 (左上方，稍微偏前) ==========
    # 角度调整：更陡峭的角度产生更明显的阴影
    key_angle = (55, 15, -50)  # 更高的仰角，更明显的阴影方向
    key_x = key_angle[0] * np.pi / 180
    key_y = key_angle[1] * np.pi / 180
    key_z = key_angle[2] * np.pi / 180

    bpy.ops.object.light_add(type='SUN', rotation=(key_x, key_y, key_z))
    key_light = bpy.context.object
    key_light.name = "Key_Light"
    key_light.data.use_nodes = True
    key_light.data.energy = key_strength * 1.8  # 增强主光强度
    key_light.data.angle = 0.08  # 更锐利的阴影边缘
    # 中性偏暖的光色，避免过白
    key_light.data.color = (1.0, 0.96, 0.92)

    # ========== Fill Light - 补光 (右侧，较柔和) ==========
    fill_angle = (35, -5, 120)
    fill_x = fill_angle[0] * np.pi / 180
    fill_y = fill_angle[1] * np.pi / 180
    fill_z = fill_angle[2] * np.pi / 180

    bpy.ops.object.light_add(type='SUN', rotation=(fill_x, fill_y, fill_z))
    fill_light = bpy.context.object
    fill_light.name = "Fill_Light"
    fill_light.data.use_nodes = True
    fill_light.data.energy = key_strength * fill_ratio * 1.5  # 增强补光
    fill_light.data.angle = 0.20  # 柔和阴影
    # 冷色调补光，增加层次感
    fill_light.data.color = (0.92, 0.95, 1.0)

    # ========== Rim/Back Light - 轮廓光 (从后方) ==========
    rim_angle = (20, 0, 165)
    rim_x = rim_angle[0] * np.pi / 180
    rim_y = rim_angle[1] * np.pi / 180
    rim_z = rim_angle[2] * np.pi / 180

    bpy.ops.object.light_add(type='SUN', rotation=(rim_x, rim_y, rim_z))
    rim_light = bpy.context.object
    rim_light.name = "Rim_Light"
    rim_light.data.use_nodes = True
    rim_light.data.energy = key_strength * rim_ratio * 1.2  # 增强轮廓光
    rim_light.data.angle = 0.05  # 锐利边缘
    rim_light.data.color = (1.0, 1.0, 1.0)

    # ========== 顶部补光 - 增加整体亮度但不增加阴影 ==========
    top_angle = (85, 0, 0)  # 几乎垂直向下
    top_x = top_angle[0] * np.pi / 180
    top_y = top_angle[1] * np.pi / 180
    top_z = top_angle[2] * np.pi / 180

    bpy.ops.object.light_add(type='SUN', rotation=(top_x, top_y, top_z))
    top_light = bpy.context.object
    top_light.name = "Top_Fill_Light"
    top_light.data.use_nodes = True
    top_light.data.energy = key_strength * 0.4  # 柔和的顶部补光
    top_light.data.angle = 0.5  # 非常柔和，几乎无阴影
    top_light.data.color = (1.0, 1.0, 1.0)

    # ========== Ambient light - 环境光增强整体亮度 ==========
    setLight_ambient(color=(0.08, 0.08, 0.09, 1))  # 增强环境光

    return key_light, fill_light, rim_light


def build_shadow_catcher_ground(translation=(0, 0, 0), plane_size=20.0):
    """
    Create a shadow catcher ground plane for clean SIGGRAPH-style renders.
    The ground is invisible but catches shadows.
    """
    bpy.ops.mesh.primitive_plane_add(size=plane_size, location=translation)
    ground = bpy.context.object
    ground.name = "Shadow_Catcher_Ground"

    # Make it a shadow catcher
    try:
        ground.is_shadow_catcher = True  # Blender 3.x+
    except:
        ground.cycles.is_shadow_catcher = True  # Blender 2.x

    # Create a simple material
    mat = bpy.data.materials.new(name="Shadow_Catcher_Mat")
    mat.use_nodes = True

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new(type='ShaderNodeOutputMaterial')
    diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
    diffuse.inputs['Color'].default_value = (1, 1, 1, 1)

    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])

    ground.data.materials.append(mat)

    return ground


def set_siggraph_render_settings(scene, camera, num_samples=512, use_high_quality=True):
    """
    Configure Cycles renderer for SIGGRAPH-quality output.
    优化版本：更亮的整体效果，避免过曝。
    """
    scene.camera = camera
    scene.render.engine = 'CYCLES'

    # High quality sampling
    scene.cycles.samples = num_samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01

    # Denoising
    scene.view_layers[0].cycles.use_denoising = True

    # Transparent background for compositing
    scene.render.film_transparent = True

    # ========== 色彩管理 - 关键设置避免过曝 ==========
    scene.view_settings.view_transform = 'Filmic'  # Filmic防止高光过曝
    scene.view_settings.look = 'High Contrast'  # 更高对比度，阴影更明显
    scene.view_settings.exposure = 0.3  # 稍微增加曝光提亮整体
    scene.view_settings.gamma = 1.0

    # Filmic的特点是高光压缩，即使增加曝光也不容易过曝
    # 如果需要更亮，可以继续增加exposure到0.5

    # GPU acceleration
    try:
        scene.cycles.device = "GPU"
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            d["use"] = 1
    except:
        print("GPU not available, using CPU")

    # Motion blur off for static figures
    scene.render.use_motion_blur = False

    if use_high_quality:
        # Higher bounces for better global illumination
        scene.cycles.max_bounces = 12
        scene.cycles.diffuse_bounces = 6  # 增加漫反射弹射，更好的全局光照
        scene.cycles.glossy_bounces = 4
        scene.cycles.transparent_max_bounces = 8
        scene.cycles.transmission_bounces = 4


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


def shift_action_frames(action, offset):
    """
    Shift all keyframes in an action by offset frames.

    Args:
        action: The Blender action to modify
        offset: Number of frames to shift (positive = forward, negative = backward)
    """
    for fcurve in action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.co.x += offset
            keyframe.handle_left.x += offset
            keyframe.handle_right.x += offset


def clear_xy_translation(action, debug=True):
    """
    Clear X and Y translation from the root bone only, keeping Z (height).
    This removes horizontal movement while preserving vertical motion and all other bone positions.
    Uses the first keyframe's value as the fixed position for all frames.

    Args:
        action: The Blender action to modify
        debug: Print debug information about fcurves
    """
    # Common root bone names in different skeleton formats
    root_bone_names = ['root', 'Root', 'ROOT', 'Hips', 'hips', 'pelvis', 'Pelvis',
                       'hip', 'Hip', 'Bip01', 'mixamorig:Hips', 'Armature',
                       'f_avg_root', 'm_avg_root']  # SMPL root bones

    if debug:
        print(f"=== Analyzing action: {action.name} ===")
        print("All fcurves with 'location' in data_path:")
        for fcurve in action.fcurves:
            if 'location' in fcurve.data_path.lower() or fcurve.data_path == 'location':
                print(f"  - {fcurve.data_path} [index={fcurve.array_index}]")

    cleared_count = 0
    for fcurve in action.fcurves:
        data_path = fcurve.data_path

        # Check if this is a location fcurve for X or Y axis (index 0 or 1)
        # Z axis (index 2) is preserved for height
        if fcurve.array_index not in [0, 1]:
            continue

        should_clear = False

        # Case 1: Armature object level location - this is root movement
        if data_path == 'location':
            should_clear = True

        # Case 2: Root bone location only (pose.bones["RootBone"].location)
        elif data_path.endswith('.location') and 'pose.bones[' in data_path:
            # Only clear if it's a known root bone
            for bone_name in root_bone_names:
                if f'pose.bones["{bone_name}"]' in data_path or f"pose.bones['{bone_name}']" in data_path:
                    should_clear = True
                    break

        if should_clear and len(fcurve.keyframe_points) > 0:
            # Use the first keyframe's value as the fixed position
            first_value = fcurve.keyframe_points[0].co.y
            print(f"Fixing XY translation for: {data_path} index={fcurve.array_index}, fixed to {first_value}")
            for keyframe in fcurve.keyframe_points:
                keyframe.co.y = first_value  # Keep the first frame's position
                keyframe.handle_left.y = first_value
                keyframe.handle_right.y = first_value
            cleared_count += 1

    print(f"Fixed {cleared_count} fcurves for XY translation")


def load_fbx_at_frame(fbx_path, frame, x_offset, instance_id, material_index=0, target_frame=1,
                      palette='academic', use_siggraph_material=True, use_sss=True):
    """
    Load FBX file and shift animation so that the specified frame becomes target_frame.

    Args:
        fbx_path: Path to the FBX file
        frame: The original frame number to extract
        x_offset: X-axis offset for positioning
        instance_id: Unique ID for naming
        material_index: Index for material color
        target_frame: The frame number where the specified frame should appear (default: 1)
        palette: SIGGRAPH color palette name ('academic', 'minimal', 'contrast', 'clay', 'professional')
        use_siggraph_material: Whether to use SIGGRAPH-quality material settings
        use_sss: Whether to use subsurface scattering

    Returns:
        Tuple of (armature, mesh_object_list)
    """
    # Track objects before import
    keys_old = set(bpy.data.objects.keys())

    # Import FBX
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    keys_new = set(bpy.data.objects.keys())
    obj_names = list(keys_new - keys_old)


    # Find armature and mesh objects
    armature, mesh_object, mesh_object_list = find_armature_and_mesh(obj_names)
    armature.location.x += x_offset

    # Shift animation so that 'frame' becomes 'target_frame'
    if armature and armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        # Calculate offset: we want frame -> target_frame, so offset = target_frame - frame
        offset = target_frame - frame
        shift_action_frames(action, offset)
        # Clear X and Y translation, keep only Z (height)
        clear_xy_translation(action)

    # Rename objects to avoid conflicts
    for obj_name in obj_names:
        obj = bpy.data.objects[obj_name]
        obj.name = f"{obj_name}_inst{instance_id}_f{frame}"

    # Set the scene to target_frame to display the desired pose
    # bpy.context.scene.frame_set(target_frame)

    # Get color from SIGGRAPH palette
    colors = SIGGRAPH_PALETTES.get(palette, SIGGRAPH_PALETTES['academic'])
    color_idx = material_index % len(colors)
    color = colors[color_idx]

    # Set material for mesh objects
    for mesh_obj in mesh_object_list:
        if use_siggraph_material:
            # Use SIGGRAPH-quality material with subtle SSS
            set_siggraph_material(mesh_obj, color, use_sss=use_sss, sss_weight=0.03)
        else:
            # Fallback to original material
            if len(mesh_obj.data.materials) == 0:
                matname = f"Material_inst{instance_id}_f{frame}"
                mat = add_material(matname, use_nodes=True, make_node_tree_empty=False)
                mesh_obj.data.materials.append(mat)
                set_material_i(bpy.data.materials[matname], material_index, use_plastic=False)

    return armature, mesh_object_list


def sample_frames(frame_start, frame_end, num_frames=8, frame_indices=None):
    """
    Get the frame indices to sample.

    Args:
        frame_start: Start frame of animation
        frame_end: End frame of animation
        num_frames: Number of frames to sample (default 8)
        frame_indices: Manual list of frame indices (overrides num_frames)

    Returns:
        List of frame indices
    """
    if frame_indices is not None:
        # Use manually specified frame indices
        return [int(f) for f in frame_indices]
    else:
        # Uniformly sample frames
        total_frames = frame_end - frame_start + 1
        if num_frames >= total_frames:
            return list(range(frame_start, frame_end + 1))
        else:
            # Sample uniformly including first and last frames
            indices = np.linspace(frame_start, frame_end, num_frames, dtype=int)
            return indices.tolist()


def get_animation_range(fbx_path):
    """
    Load FBX temporarily to get animation frame range, then delete it.

    Returns:
        Tuple of (frame_start, frame_end)
    """
    # Track objects before import
    keys_old = set(bpy.data.objects.keys())

    # Import FBX
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    keys_new = set(bpy.data.objects.keys())
    obj_names = list(keys_new - keys_old)

    # Find armature
    armature, _, _ = find_armature_and_mesh(obj_names)

    # Get frame range
    frame_start = bpy.context.scene.frame_start
    frame_end = bpy.context.scene.frame_end

    if armature and armature.animation_data and armature.animation_data.action:
        action = armature.animation_data.action
        frame_start = int(action.frame_range[0])
        frame_end = int(action.frame_range[1])

    # Delete the temporarily imported objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj_name in obj_names:
        if obj_name in bpy.data.objects:
            bpy.data.objects[obj_name].select_set(True)
    bpy.ops.object.delete()

    return frame_start, frame_end


if __name__ == '__main__':
    # ${blender} -noaudio --python examples/render_smpl_shot_ground.py -- test_baseline.fbx --num_frames 8 --material_index 0
    # SIGGRAPH quality: add --siggraph flag for professional paper figure rendering
    parser = get_parser()
    parser.add_argument('--num_frames', type=int, default=6,
                        help='Number of frames to sample and visualize (default: 6)')
    parser.add_argument('--skip_start', type=int, default=10,
                        help='Number of frames to skip at start (default: 10)')
    parser.add_argument('--skip_end', type=int, default=10,
                        help='Number of frames to skip at end (default: 10)')

    parser.add_argument('--frame_indices', '--frames', type=int, nargs='+', default=None,
                        help='Manual list of frame indices to visualize (overrides --num_frames), e.g. --frames 1 10 20 30')
    parser.add_argument('--spacing', type=float, default=1.0,
                        help='Spacing between frames along x-axis (default: 1.0)')
    parser.add_argument('--material_index', type=int, default=0,
                        help='Material color index (default: 0)')

    # SIGGRAPH-quality rendering options
    parser.add_argument('--siggraph', action='store_true',
                        help='Enable SIGGRAPH-quality rendering (better lighting, materials, colors)')
    parser.add_argument('--palette', type=str, default='academic',
                        choices=['academic', 'minimal', 'contrast', 'clay', 'professional'],
                        help='SIGGRAPH color palette (default: academic)')
    parser.add_argument('--no_sss', action='store_true',
                        help='Disable subsurface scattering (faster rendering)')
    parser.add_argument('--key_light_strength', type=float, default=1.0,
                        help='Key light strength for SIGGRAPH lighting (default: 4.0)')
    parser.add_argument('--shadow_catcher', action='store_true',
                        help='Use shadow catcher ground instead of checkerboard')
    parser.add_argument('--high_quality', action='store_true',
                        help='Enable highest quality settings (more samples, higher bounces)')

    args = parse_args(parser)

    # Setup scene with neutral background
    setup(rgb=(0.95, 0.95, 0.95, 1) if args.siggraph else (1, 1, 1, 1))

    # Load the FBX file
    fbx_path = args.path
    assert fbx_path.endswith('.fbx') or fbx_path.endswith('.FBX'), \
        f"Input file must be an FBX file, got: {fbx_path}"

    print(f"Loading FBX file: {fbx_path}")
    print(f"SIGGRAPH mode: {'ON' if args.siggraph else 'OFF'}")
    if args.siggraph:
        print(f"  Palette: {args.palette}")
        print(f"  SSS: {'OFF' if args.no_sss else 'ON'}")
        print(f"  High Quality: {'ON' if args.high_quality else 'OFF'}")

    # First, get animation frame range
    frame_start, frame_end = get_animation_range(fbx_path)
    frame_start += args.skip_start
    frame_end -= args.skip_end
    print(f"Animation frame range: {frame_start} to {frame_end}")

    # Get frames to sample
    frames_to_sample = sample_frames(
        frame_start, frame_end,
        num_frames=args.num_frames,
        frame_indices=args.frame_indices
    )
    print(f"Sampling frames: {frames_to_sample}")

    # Calculate x positions (centered around 0)
    num_samples = len(frames_to_sample)
    x_positions = [(i - (num_samples - 1) / 2) * args.spacing for i in range(num_samples)]

    # Load FBX for each sampled frame
    all_armatures = []
    all_meshes = []
    for i, (frame, x_pos) in enumerate(zip(frames_to_sample, x_positions)):
        print(f"Loading instance {i+1}/{num_samples}: frame {frame} at x={x_pos:.2f}")
        armature, mesh_list = load_fbx_at_frame(
            fbx_path, frame, x_pos,
            instance_id=i,
            material_index=args.material_index,
            palette=args.palette,
            use_siggraph_material=False,
            use_sss=not args.no_sss
        )
        if armature:
            all_armatures.append(armature)
        all_meshes.extend(mesh_list)

    # Calculate camera position based on the arrangement
    total_width = (num_samples - 1) * args.spacing

    # Adjust camera distance based on number of samples
    camera_distance = 15 + num_samples * 2
    camera_height = 1.8

    camera = set_camera(
        location=(0, -camera_distance, camera_height),
        center=(0, 0, 0.9),
        focal=85 if args.siggraph else 100  # Slightly wider FOV for SIGGRAPH
    )

    # Setup lighting
    if args.siggraph:
        # SIGGRAPH three-point lighting
        print("Setting up SIGGRAPH three-point lighting...")
        setup_siggraph_lighting(
            num_samples_width=total_width,
            key_strength=args.key_light_strength,
            fill_ratio=0.35,
            rim_ratio=0.45
        )
    else:
        # Original simple lighting
        lightAngle = [45, 45, 0]
        strength = 4
        shadowSoftness = 0.3
        sun = setLight_sun(lightAngle, strength, shadowSoftness)
        setLight_ambient(color=(0.1, 0.1, 0.1, 1))

    # Build ground plane
    ground_size = int(num_samples * args.spacing * 1.2)
    if args.shadow_catcher:
        # Shadow catcher for clean compositing
        build_shadow_catcher_ground(translation=(0, ground_size//2-1, 0), plane_size=ground_size)
    else:
        # Standard checkerboard ground
        build_plane(translation=(0, ground_size//2-1, 0), plane_size=ground_size)

    # Setup renderer
    render_samples = 512

    if True:
        set_siggraph_render_settings(
            bpy.context.scene,
            bpy.data.objects["Camera"],
            num_samples=render_samples,
            use_high_quality=True
        )
    else:
        set_cycles_renderer(
            bpy.context.scene,
            bpy.data.objects["Camera"],
            num_samples=render_samples,
            use_transparent_bg=True,
            use_denoising=True,
        )

    # Set output properties - SIGGRAPH figures typically need high resolution
    outdir = args.out
    res_x = 2048
    res_y = 720

    set_output_properties(
        bpy.context.scene,
        output_file_path=outdir,
        res_x=res_x,
        res_y=res_y,
        tile_x=res_x,
        tile_y=res_y,
        resolution_percentage=100,
        format='PNG'
    )

    # Set to frame 1 for single frame render
    bpy.context.scene.frame_set(1)
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 1

    # Save blend file if specified
    if args.out_blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.out_blend)
        print(f"Saved blend file to: {args.out_blend}")

    print(f"\n{'='*60}")
    print(f"Scene setup complete.")
    print(f"Output: {outdir}")
    print(f"Resolution: {res_x}x{res_y}")
    print(f"Samples: {render_samples}")
    print(f"Sampled {num_samples} frames: {frames_to_sample}")
    if args.siggraph:
        print(f"SIGGRAPH Quality: Palette={args.palette}, SSS={'OFF' if args.no_sss else 'ON'}")
    print(f"{'='*60}\n")

    bpy.ops.render.render(write_still=True)
