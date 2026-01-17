import os
import math
import time
import bpy
from myblender.setup import (
    setLight_sun,
    setLight_ambient,
    get_parser,
    parse_args,
    set_cycles_renderer,
    set_output_properties,
    setup,
    render_with_progress,
)
from myblender.material import set_material_i, add_material
from myblender.geometry import (
    set_camera,
    build_plane
)

def setup_mist_fog(scene, start=5.0, depth=20.0, fog_color=(0.7, 0.8, 0.9)):
    """
    修正后的雾气设置：
    1. 增强了节点连接的鲁棒性（适配新旧版本Blender）。
    2. 移除了 Set Alpha 节点，确保雾气能覆盖背景。
    3. 默认颜色稍微调深一点蓝色，以便在白色背景下能看清。
    """
    # 1. 开启 Mist Pass
    scene.view_layers[0].use_pass_mist = True

    # 2. 配置世界 Mist 参数
    scene.world.mist_settings.use_mist = True
    scene.world.mist_settings.start = start
    scene.world.mist_settings.depth = depth
    scene.world.mist_settings.falloff = 'QUADRATIC'

    # 3. 设置合成节点
    scene.use_nodes = True
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    nodes.clear()

    # --- 节点创建 ---

    # 渲染层输入
    render_layers = nodes.new(type='CompositorNodeRLayers')
    render_layers.location = (-300, 0)

    # 雾的颜色 (稍微带点蓝灰，增加对比度)
    fog_color_node = nodes.new(type='CompositorNodeRGB')
    fog_color_node.location = (-100, 200)
    fog_color_node.outputs[0].default_value = (*fog_color, 1.0)

    # Mix 节点 (核心)
    # 自动适配 Blender 版本差异
    try:
        # Blender 3.4+ / 4.0+
        mix = nodes.new(type='CompositorNodeMix')
        mix.data_type = 'RGBA'
        mix.blend_type = 'MIX'
        # 新版 Mix 节点的输入索引：0:Factor, 1:A, 2:B (如果类型是RGBA)
        # 我们使用索引连接比使用名称更安全
        input_fac = 0
        input_image_a = 6 # 在某些版本RGBA模式下，A是6，B是7，或者直接按顺序
        # 为了最安全，我们下面通过连线逻辑动态判断，或者使用 MixRGB (旧节点兼容性最好)
    except:
        pass

    # 为了保证绝对兼容性，推荐直接使用 MixRGB (在Python API中依然可用且稳定)
    mix = nodes.new(type='CompositorNodeMixRGB')
    mix.blend_type = 'MIX'
    mix.location = (100, 0)

    # 输出节点
    composite = nodes.new(type='CompositorNodeComposite')
    composite.location = (400, 0)

    # --- 节点连接 ---

    # 逻辑：
    # Factor = Mist Pass (距离越远，值越大)
    # Input 1 (Top) = 原始图像 (近处清晰)
    # Input 2 (Bottom) = 雾颜色 (远处全是雾)

    # 1. Mist 连接到 Factor
    links.new(render_layers.outputs['Mist'], mix.inputs['Fac'])

    # 2. 原始图像 连接到 Image 1
    links.new(render_layers.outputs['Image'], mix.inputs[1])

    # 3. 雾颜色 连接到 Image 2
    links.new(fog_color_node.outputs['RGBA'], mix.inputs[2])

    # 4. 结果直接输出 (不要再 Set Alpha，让雾充满背景)
    links.new(mix.outputs['Image'], composite.inputs['Image'])

    print(f"Fog setup complete. Start: {start}, Depth: {depth}")

def create_studio_backdrop(center=(0, 0, 0), width=20, depth=15, height=10,
                           bevel_radius=3.0, color=(0.95, 0.95, 0.95, 1.0)):
    """
    Create an L-shaped seamless studio backdrop (cyclorama).

    Args:
        center: Center position of the backdrop
        width: Width of the backdrop (X direction)
        depth: Depth of the backdrop (Y direction)
        height: Height of the back wall
        bevel_radius: Radius of the curved transition between floor and wall
        color: Background color (RGBA)
    """
    import bmesh

    # Create a new mesh
    mesh = bpy.data.meshes.new("StudioBackdrop")
    obj = bpy.data.objects.new("StudioBackdrop", mesh)
    bpy.context.collection.objects.link(obj)

    # Create BMesh for easier mesh manipulation
    bm = bmesh.new()

    # Create the floor plane vertices
    # Floor extends from center forward (positive Y) and to both sides (X)
    half_width = width / 2

    # Floor vertices (at z=0)
    v_floor_fl = bm.verts.new((center[0] - half_width, center[1] + depth, 0))  # front left
    v_floor_fr = bm.verts.new((center[0] + half_width, center[1] + depth, 0))  # front right
    v_floor_bl = bm.verts.new((center[0] - half_width, center[1] - depth + bevel_radius, 0))  # back left (before curve)
    v_floor_br = bm.verts.new((center[0] + half_width, center[1] - depth + bevel_radius, 0))  # back right (before curve)

    # Create floor face
    bm.faces.new([v_floor_fl, v_floor_fr, v_floor_br, v_floor_bl])

    # Create curved transition (bevel) using multiple segments
    num_segments = 16
    curve_verts_left = [v_floor_bl]
    curve_verts_right = [v_floor_br]

    for i in range(1, num_segments + 1):
        angle = (math.pi / 2) * (i / num_segments)  # 0 to 90 degrees
        # Parametric circle: center at (center[1] - depth + bevel_radius, bevel_radius)
        y = center[1] - depth + bevel_radius - bevel_radius * math.sin(angle)
        z = bevel_radius - bevel_radius * math.cos(angle)

        v_left = bm.verts.new((center[0] - half_width, y, z))
        v_right = bm.verts.new((center[0] + half_width, y, z))
        curve_verts_left.append(v_left)
        curve_verts_right.append(v_right)

    # Create faces for the curved section
    for i in range(len(curve_verts_left) - 1):
        bm.faces.new([
            curve_verts_left[i], curve_verts_right[i],
            curve_verts_right[i + 1], curve_verts_left[i + 1]
        ])

    # Create the back wall (from top of curve to full height)
    v_wall_tl = bm.verts.new((center[0] - half_width, center[1] - depth, height))
    v_wall_tr = bm.verts.new((center[0] + half_width, center[1] - depth, height))

    # Wall face (from curve top to wall top)
    bm.faces.new([
        curve_verts_left[-1], curve_verts_right[-1],
        v_wall_tr, v_wall_tl
    ])

    # Update mesh
    bm.to_mesh(mesh)
    bm.free()

    # Smooth shading for the curved part
    for poly in mesh.polygons:
        poly.use_smooth = True

    # Create material
    mat = bpy.data.materials.new(name="StudioBackdropMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)

    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (100, 0)
    principled.inputs['Base Color'].default_value = color
    principled.inputs['Roughness'].default_value = 0.5  # Matte finish
    principled.inputs['Metallic'].default_value = 0.0

    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    obj.data.materials.append(mat)

    print(f"Studio backdrop created: {width}x{depth}x{height}m, bevel={bevel_radius}m")
    return obj


def set_transparent_ghost_material(mesh_obj, progress, matname, alpha=0.3):
    """
    Set a semi-transparent ghost-like material for virtual/predicted characters.

    Args:
        mesh_obj: The mesh object to apply material to
        progress: Value from 0 to 1 for gradient color
        matname: Name for the material
        alpha: Transparency level (0=fully transparent, 1=opaque)
    """
    # Use a lighter, more ethereal color palette for ghost effect
    # light_color = (0.6, 0.75, 0.9, 1.0)   # Light ethereal blue
    light_color = (0.5, 0.6, 0.9, 1.0)   # Light ethereal blue
    dark_color = (0.3, 0.5, 0.75, 1.0)    # Slightly darker blue

    # Interpolate color based on progress
    r = light_color[0] + (dark_color[0] - light_color[0]) * progress
    g = light_color[1] + (dark_color[1] - light_color[1]) * progress
    b = light_color[2] + (dark_color[2] - light_color[2]) * progress

    base_color = (r, g, b, 1.0)

    # Create material with transparency
    mat = bpy.data.materials.new(name=matname)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'  # Enable alpha blending
    mat.shadow_method = 'HASHED'  # Better shadows for transparent objects
    mat.use_backface_culling = False

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (600, 0)

    # Principled BSDF
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (300, 0)
    principled.inputs['Base Color'].default_value = base_color
    principled.inputs['Alpha'].default_value = alpha
    principled.inputs['Roughness'].default_value = 0.4
    principled.inputs['Metallic'].default_value = 0.0

    # Add subtle emission for ethereal glow
    try:
        principled.inputs['Emission Color'].default_value = (r * 0.3, g * 0.3, b * 0.4, 1.0)
        principled.inputs['Emission Strength'].default_value = 0.2
    except KeyError:
        try:
            principled.inputs['Emission'].default_value = (r * 0.3, g * 0.3, b * 0.4, 1.0)
        except KeyError:
            pass

    # Subsurface for soft ghostly look
    try:
        principled.inputs['Subsurface Weight'].default_value = 0.1
    except KeyError:
        try:
            principled.inputs['Subsurface'].default_value = 0.1
        except KeyError:
            pass

    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    # Clear existing materials and apply new one
    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(mat)

    print(f"Ghost material applied: {matname}, alpha={alpha}, progress={progress:.2f}")
    return mat


def set_gradient_blue_material(mesh_obj, progress, matname):
    """
    Set a high-quality clay-like gradient blue material with enhanced AO.

    Features:
    - Ambient Occlusion for enhanced muscle/joint shadows
    - Clay-like subsurface scattering for volume
    - Subtle specular highlights for sculptural look

    Args:
        mesh_obj: The mesh object to apply material to
        progress: Value from 0 (light blue) to 1 (dark blue)
        matname: Name for the material
    """
    # Light blue (start) to Dark blue (end) - slightly desaturated for clay look
    light_blue = (0.55, 0.72, 0.88, 1.0)
    dark_blue = (0.12, 0.28, 0.52, 1.0)

    # Interpolate between light and dark blue
    r = light_blue[0] + (dark_blue[0] - light_blue[0]) * progress
    g = light_blue[1] + (dark_blue[1] - light_blue[1]) * progress
    b = light_blue[2] + (dark_blue[2] - light_blue[2]) * progress

    base_color = (r, g, b, 1.0)

    # Darker version for AO shadows
    ao_shadow_color = (r * 0.4, g * 0.4, b * 0.5, 1.0)

    # Create material
    mat = bpy.data.materials.new(name=matname)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # =========================================================
    # Create nodes for clay-like material with AO
    # =========================================================

    # Output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (800, 0)

    # Principled BSDF - main shader
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (500, 0)

    # Ambient Occlusion node - for enhanced cavity shadows
    ao_node = nodes.new(type='ShaderNodeAmbientOcclusion')
    ao_node.location = (-200, 100)
    ao_node.samples = 16
    ao_node.inputs['Distance'].default_value = 0.15  # Small distance for fine details

    # Base color node
    base_color_node = nodes.new(type='ShaderNodeRGB')
    base_color_node.location = (-400, 200)
    base_color_node.outputs[0].default_value = base_color

    # Shadow color node (darker for AO cavities)
    shadow_color_node = nodes.new(type='ShaderNodeRGB')
    shadow_color_node.location = (-400, 0)
    shadow_color_node.outputs[0].default_value = ao_shadow_color

    # Mix node - blend base and shadow colors based on AO
    # Use ShaderNodeMix for Blender 3.4+, fallback to ShaderNodeMixRGB for older versions
    try:
        mix_color = nodes.new(type='ShaderNodeMix')
        mix_color.data_type = 'RGBA'
        mix_color.blend_type = 'MIX'
        use_new_mix = True
    except:
        mix_color = nodes.new(type='ShaderNodeMixRGB')
        mix_color.blend_type = 'MIX'
        use_new_mix = False
    mix_color.location = (100, 100)

    # Color Ramp to control AO intensity/contrast
    color_ramp = nodes.new(type='ShaderNodeValToRGB')
    color_ramp.location = (-50, -100)
    # Adjust ramp for stronger AO effect
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)  # Black (shadow)
    color_ramp.color_ramp.elements[1].position = 0.7  # Sharper transition
    color_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)  # White (lit)

    # Fresnel node for subtle rim lighting effect
    fresnel = nodes.new(type='ShaderNodeFresnel')
    fresnel.location = (100, -200)
    fresnel.inputs['IOR'].default_value = 1.45  # Subtle fresnel

    # =========================================================
    # Connect nodes
    # =========================================================

    # AO -> Color Ramp (for contrast control)
    links.new(ao_node.outputs['AO'], color_ramp.inputs['Fac'])

    # Connect Mix node based on version
    if use_new_mix:
        # ShaderNodeMix (Blender 3.4+) - RGBA mode
        # Inputs: 0=Factor, 6=A (RGBA), 7=B (RGBA)
        # Outputs: 2=Result (RGBA)
        links.new(color_ramp.outputs['Color'], mix_color.inputs['Factor'])
        links.new(shadow_color_node.outputs['Color'], mix_color.inputs[6])  # A input
        links.new(base_color_node.outputs['Color'], mix_color.inputs[7])  # B input
        links.new(mix_color.outputs[2], principled.inputs['Base Color'])
    else:
        # ShaderNodeMixRGB (older Blender)
        # Inputs: 0=Fac, 1=Color1, 2=Color2
        # Outputs: 0=Color
        links.new(color_ramp.outputs['Color'], mix_color.inputs['Fac'])
        links.new(shadow_color_node.outputs['Color'], mix_color.inputs[1])  # Color1
        links.new(base_color_node.outputs['Color'], mix_color.inputs[2])  # Color2
        links.new(mix_color.outputs['Color'], principled.inputs['Base Color'])

    # Connect Principled to Output
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    # =========================================================
    # Set Principled BSDF properties for clay look
    # =========================================================

    # Roughness: Clay is slightly rough but with some sheen
    principled.inputs['Roughness'].default_value = 0.35

    # Metallic: Non-metallic for clay
    principled.inputs['Metallic'].default_value = 0.0

    # Specular: Moderate for sculptural highlights
    try:
        principled.inputs['Specular IOR Level'].default_value = 0.5
    except KeyError:
        try:
            principled.inputs['Specular'].default_value = 0.5
        except KeyError:
            pass

    # Subsurface Scattering: Very subtle for clay volume feel
    # IMPORTANT: Set SSS color to match base color to avoid red color bleeding
    try:
        # Blender 4.0+ API
        principled.inputs['Subsurface Weight'].default_value = 0.06
        principled.inputs['Subsurface Scale'].default_value = 0.08
        # Set radius to equal values for uniform scattering (no color tint)
        principled.inputs['Subsurface Radius'].default_value = (0.4, 0.4, 0.4)
    except KeyError:
        try:
            # Older Blender API
            principled.inputs['Subsurface'].default_value = 0.04
            # Set subsurface color to match base color (critical to avoid red tint!)
            principled.inputs['Subsurface Color'].default_value = base_color
            # Equal radius values prevent color shifting
            principled.inputs['Subsurface Radius'].default_value = (0.3, 0.3, 0.3)
        except KeyError:
            pass

    # Sheen: Subtle sheen for clay-like appearance
    try:
        principled.inputs['Sheen Weight'].default_value = 0.1
    except KeyError:
        try:
            principled.inputs['Sheen'].default_value = 0.1
        except KeyError:
            pass

    # Coat: Very subtle clearcoat for sculptural look
    try:
        principled.inputs['Coat Weight'].default_value = 0.05
        principled.inputs['Coat Roughness'].default_value = 0.3
    except KeyError:
        pass

    # Clear existing materials and apply new one
    mesh_obj.data.materials.clear()
    mesh_obj.data.materials.append(mat)

    print(f"Clay material applied: {matname}, progress={progress:.2f}")
    return mat


def setup_studio_three_point_lighting(center=(0, 0, 0), key_strength=800.0):
    """
    Setup professional balanced lighting for studio renders.
    Even lighting from both sides to avoid dark/bright imbalance.
    All lights use track_to constraint to aim at the subject center.

    Args:
        center: Scene center point (where the subject is located)
        key_strength: Main light intensity (for Area lights, typically 200-1000)
    """
    # Create an empty at the target center for lights to track
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(center[0], center[1], center[2]))
    light_target = bpy.context.object
    light_target.name = "Light_Target"

    def add_track_to_constraint(light_obj, target_obj):
        """Add track-to constraint so light always aims at target"""
        constraint = light_obj.constraints.new(type='TRACK_TO')
        constraint.target = target_obj
        constraint.track_axis = 'TRACK_NEGATIVE_Z'  # Area lights emit from -Z
        constraint.up_axis = 'UP_Y'

    # =========================================================
    # Left Key Light - Large soft Area Light from front-left
    # =========================================================
    bpy.ops.object.light_add(type='AREA', location=(center[0] - 5, center[1] + 4, center[2] + 4))
    left_key = bpy.context.object
    left_key.name = "Left_Key_Light"
    left_key.data.energy = key_strength
    left_key.data.size = 6.0  # Large size for soft shadows
    left_key.data.color = (1.0, 0.99, 0.97)  # Slightly warm
    add_track_to_constraint(left_key, light_target)

    # =========================================================
    # Right Key Light - Balanced light from front-right (same intensity)
    # =========================================================
    bpy.ops.object.light_add(type='AREA', location=(center[0] + 5, center[1] + 4, center[2] + 4))
    right_key = bpy.context.object
    right_key.name = "Right_Key_Light"
    right_key.data.energy = key_strength  # Same as left for balance
    right_key.data.size = 6.0  # Same size for consistency
    right_key.data.color = (1.0, 0.99, 0.97)  # Same color
    add_track_to_constraint(right_key, light_target)

    # =========================================================
    # Front Fill Light - Centered front light for even illumination
    # =========================================================
    if False:
        bpy.ops.object.light_add(type='AREA', location=(center[0], center[1] + 6, center[2] + 2))
        front_fill = bpy.context.object
        front_fill.name = "Front_Fill_Light"
        front_fill.data.energy = key_strength * 0.5  # 50% fill (increased from 40%)
        front_fill.data.size = 10.0  # Very large for soft even fill
    else:
        bpy.ops.object.light_add(type='SUN', location=(center[0], center[1] + 6, center[2] + 2))
        front_fill = bpy.context.object
        front_fill.name = "Front_Fill_Light"
        front_fill.data.energy = 1.  # 50% fill (increased from 40%)
    front_fill.data.color = (1.0, 1.0, 1.0)  # Pure white
    add_track_to_constraint(front_fill, light_target)

    # =========================================================
    # Rim/Back Light - Subtle edge light for separation
    # =========================================================
    bpy.ops.object.light_add(type='AREA', location=(center[0], center[1] - 4, center[2] + 3))
    rim_light = bpy.context.object
    rim_light.name = "Rim_Light"
    rim_light.data.energy = key_strength * 0.3  # Rim light for edge definition
    rim_light.data.size = 5.0
    rim_light.data.color = (1.0, 1.0, 1.0)  # Pure white
    add_track_to_constraint(rim_light, light_target)

    # =========================================================
    # Top Light - Even overhead illumination
    # =========================================================
    bpy.ops.object.light_add(type='AREA', location=(center[0], center[1], center[2] + 6))
    top_light = bpy.context.object
    top_light.name = "Top_Light"
    top_light.data.energy = key_strength * 0.4  # Moderate top light
    top_light.data.size = 12.0  # Very large for even coverage
    top_light.data.color = (1.0, 1.0, 1.0)  # Pure white
    add_track_to_constraint(top_light, light_target)

    # =========================================================
    # World/Environment - Soft ambient fill
    # =========================================================
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    bg_node = nodes.new(type='ShaderNodeBackground')
    bg_node.location = (0, 0)
    bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # Pure white
    bg_node.inputs['Strength'].default_value = 1.0  # Bright ambient for well-lit ground

    output = nodes.new(type='ShaderNodeOutputWorld')
    output.location = (200, 0)

    links.new(bg_node.outputs['Background'], output.inputs['Surface'])

    # =========================================================
    # Enable Ambient Occlusion in render settings
    # =========================================================
    scene = bpy.context.scene

    # For Cycles
    if scene.render.engine == 'CYCLES':
        # AO is automatic in Cycles path tracing, but we can enhance it
        scene.cycles.use_fast_gi = False  # Fast GI for better ambient
        try:
            scene.cycles.ao_bounces = 2
            scene.cycles.ao_bounces_render = 2
        except:
            pass

    # For Eevee (if used)
    try:
        scene.eevee.use_gtao = True  # Ground Truth AO
        scene.eevee.gtao_distance = 0.5
        scene.eevee.gtao_factor = 1.2
    except:
        pass

    print(f"Studio balanced lighting setup: key={key_strength}, symmetric left/right")
    return left_key, right_key, front_fill, rim_light, top_light


def add_reflection_to_ground(ground, roughness=0.15, metallic=0.1, specular=0.8,
                              white_color=(1.0, 1.0, 1.0, 1.0),
                              gray_color=(0.85, 0.85, 0.85, 1.0)):
    """
    Add reflective properties to an existing checkerboard ground material.
    Also sets the checkerboard colors to be bright (white and light gray).

    Args:
        ground: The ground plane object (with checkerboard material)
        roughness: Surface roughness (0=mirror, 1=matte)
        metallic: Metallic factor for reflection
        specular: Specular reflection intensity
        white_color: Color for the white squares (RGBA)
        gray_color: Color for the alternate squares (RGBA)
    """
    # Get the material from the ground
    if len(ground.data.materials) == 0:
        print("Warning: Ground has no materials")
        return

    mat = ground.data.materials[0]
    if not mat.use_nodes:
        print("Warning: Material doesn't use nodes")
        return

    nodes = mat.node_tree.nodes

    # Find and update the Checker Texture node colors
    checker_node = None
    for node in nodes:
        if node.type == 'TEX_CHECKER':
            checker_node = node
            break

    if checker_node is not None:
        # Set bright checkerboard colors: pure white and light gray
        checker_node.inputs['Color1'].default_value = white_color
        checker_node.inputs['Color2'].default_value = gray_color
        print(f"Checkerboard colors set: white={white_color[:3]}, gray={gray_color[:3]}")

    # Find the Principled BSDF node
    principled = None
    for node in nodes:
        if node.type == 'BSDF_PRINCIPLED':
            principled = node
            break

    if principled is None:
        print("Warning: No Principled BSDF node found")
        return

    # Set reflective properties while keeping the checkerboard texture
    principled.inputs['Roughness'].default_value = roughness  # Lower = more mirror-like
    principled.inputs['Metallic'].default_value = metallic

    # Set specular (handle different Blender versions)
    try:
        principled.inputs['Specular IOR Level'].default_value = specular
    except KeyError:
        try:
            principled.inputs['Specular'].default_value = specular
        except KeyError:
            pass

    # Enable shadow receiving
    ground.visible_shadow = True

    print(f"Reflection added to ground: roughness={roughness}, metallic={metallic}, specular={specular}")


def setup_bright_studio_lighting(center=(0, 0, 0), key_strength=8.0, sun_angle=0.02):
    """
    Setup bright studio lighting with clear shadows from left-back 45 degrees.

    Args:
        center: Scene center point
        key_strength: Main light intensity
        sun_angle: Sun angular diameter (smaller = sharper shadows)
    """
    # =========================================================
    # Key Light - Sun from left-back 45 degrees for clear shadow
    # =========================================================
    # Left-back 45 degrees: X negative, Y negative direction
    # Camera is at Y positive looking at center, so:
    # - "left" from camera view = X negative
    # - "back" = Y negative (behind the subject from camera view)

    bpy.ops.object.light_add(type='SUN', location=(center[0], center[1], center[2] + 10))
    sun = bpy.context.object
    sun.name = "Key_Sun_Light"
    sun.data.energy = key_strength

    # Set sun angle for sharp shadows (smaller = sharper)
    sun.data.angle = sun_angle  # Very small angle for crisp shadows

    # Rotate to shine from left-back 45 degrees
    # Rotation: X = tilt down, Z = horizontal direction
    sun.rotation_euler = (
        math.radians(45),   # Tilt down 45 degrees from vertical
        0,
        math.radians(-135)  # -135 degrees = left-back direction (225 degrees from front)
    )

    # =========================================================
    # Fill Light - Softer light from the right-front to fill shadows
    # =========================================================
    bpy.ops.object.light_add(type='AREA', location=(center[0] + 5, center[1] + 3, center[2] + 4))
    fill_light = bpy.context.object
    fill_light.name = "Fill_Light"
    fill_light.data.energy = key_strength * 0.3  # 30% of key light
    fill_light.data.size = 5.0  # Large soft light
    fill_light.rotation_euler = (math.radians(60), 0, math.radians(30))

    # =========================================================
    # Rim/Back Light - Highlight edges from behind
    # =========================================================
    bpy.ops.object.light_add(type='AREA', location=(center[0], center[1] - 4, center[2] + 3))
    rim_light = bpy.context.object
    rim_light.name = "Rim_Light"
    rim_light.data.energy = key_strength * 0.4
    rim_light.data.size = 3.0
    rim_light.rotation_euler = (math.radians(120), 0, 0)

    # =========================================================
    # Bright World/Environment lighting
    # =========================================================
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear existing nodes
    nodes.clear()

    # Create background node
    bg_node = nodes.new(type='ShaderNodeBackground')
    bg_node.location = (0, 0)
    bg_node.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)  # Pure white
    bg_node.inputs['Strength'].default_value = 0.8  # Bright ambient

    # Create output node
    output = nodes.new(type='ShaderNodeOutputWorld')
    output.location = (200, 0)

    # Connect
    links.new(bg_node.outputs['Background'], output.inputs['Surface'])

    print(f"Studio lighting setup: key={key_strength}, sharp shadows from left-back 45°")
    return sun, fill_light, rim_light




def add_root_trajectory(armature_list, start_color=(0.2, 0.6, 1.0, 1.0), end_color=(1.0, 0.3, 0.5, 1.0),
                        line_thickness=0.02, emission_strength=3.0, pelvis_height=1.0,
                        alpha=1.0, curve_name="RootTrajectory",
                        extend_start=0.6, extend_end=0.6, add_arrow=True, arrow_scale=1.0):
    """
    Add a colorful gradient trajectory line connecting the root/pelvis positions of multiple characters.
    This emphasizes "Global Motion" and "Dynamics" by showing the movement path in world space.

    Args:
        armature_list: List of (armature, mesh_object_list) tuples from load_fbx_at_frame
        start_color: RGBA color at the beginning of trajectory (default: cyan blue)
        end_color: RGBA color at the end of trajectory (default: magenta pink)
        line_thickness: Thickness of the trajectory line
        emission_strength: Emission strength for the glowing effect
        pelvis_height: Height of the pelvis/root joint from ground (default: 1.0m)
        alpha: Transparency value (0=fully transparent, 1=opaque)
        curve_name: Name for the curve object
        extend_start: How much to extend the trajectory before the first point (in meters)
        extend_end: How much to extend the trajectory after the last point (in meters)
        add_arrow: Whether to add an arrowhead at the end
        arrow_scale: Scale factor for the arrowhead size

    Returns:
        The created curve object
    """
    from mathutils import Vector

    if len(armature_list) < 2:
        print("Warning: Need at least 2 positions to create trajectory")
        return None

    # IMPORTANT: Update scene and dependency graph to ensure all armature poses are evaluated
    # This is critical for getting correct bone positions, especially for the last imported armature
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Collect root positions from each armature at the current frame
    positions = []
    for idx, (armature, mesh_object_list) in enumerate(armature_list):
        root_pos = None
        found_bone = False

        # Get the evaluated armature (with all modifiers and poses applied)
        armature_eval = armature.evaluated_get(depsgraph)

        # Try to find pelvis/hips bone for accurate position
        if armature_eval.pose and armature_eval.pose.bones:
            # Common pelvis bone names in SMPL/FBX
            pelvis_names = ['pelvis', 'Pelvis', 'mixamorig:Hips', 'Hips', 'hips',
                           'm_avg_Pelvis', 'f_avg_Pelvis', 'root', 'Root']

            # Debug: print available bone names for first armature
            if idx == 0:
                print(f"Available bones: {[b.name for b in armature_eval.pose.bones]}")

            for bone_name in pelvis_names:
                if bone_name in armature_eval.pose.bones:
                    bone = armature_eval.pose.bones[bone_name]
                    # Get bone world position using bone.matrix (includes pose)
                    # bone.matrix is in armature space, need to transform to world
                    bone_world_matrix = armature_eval.matrix_world @ bone.matrix
                    root_pos = bone_world_matrix.translation.copy()
                    found_bone = True
                    print(f"Frame {idx}: Found bone '{bone_name}' at world position {root_pos}")
                    break

        # Fallback: use armature location + pelvis height
        if not found_bone:
            armature_loc = armature.location.copy()
            root_pos = Vector((armature_loc.x, armature_loc.y, armature_loc.z + pelvis_height))
            print(f"Frame {idx}: Using armature location + pelvis_height: {root_pos}")

        positions.append(root_pos)

    # Extend trajectory at the start (before first point)
    if extend_start > 0 and len(positions) >= 2:
        start_dir = (positions[0] - positions[1]).normalized()
        extended_start = positions[0] + start_dir * extend_start
        positions.insert(0, extended_start)
        print(f"Extended trajectory start by {extend_start}m")

    # Extend trajectory at the end (after last point)
    if extend_end > 0 and len(positions) >= 2:
        end_dir = (positions[-1] - positions[-2]).normalized()
        extended_end = positions[-1] + end_dir * extend_end
        positions.append(extended_end)
        print(f"Extended trajectory end by {extend_end}m")

    # Create a curve for the trajectory
    curve_data = bpy.data.curves.new(curve_name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 12
    curve_data.bevel_depth = line_thickness
    curve_data.bevel_resolution = 4
    curve_data.fill_mode = 'FULL'

    # Create a smooth spline through all positions
    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(len(positions) - 1)

    for i, pos in enumerate(positions):
        bp = spline.bezier_points[i]
        bp.co = pos
        bp.handle_left_type = 'AUTO'
        bp.handle_right_type = 'AUTO'

    # Create the curve object
    curve_obj = bpy.data.objects.new(curve_name, curve_data)
    bpy.context.collection.objects.link(curve_obj)

    # Create gradient emission material using curve parameter
    mat_name = f"{curve_name}_Material"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    # Enable transparency if alpha < 1
    if alpha < 1.0:
        mat.blend_method = 'BLEND'
        mat.shadow_method = 'HASHED'

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (600, 0)

    # Mix Shader to combine emission with principled for nice glow effect
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    mix_shader.location = (400, 0)
    mix_shader.inputs['Fac'].default_value = 0.7  # More emission than diffuse

    # Principled BSDF for base color
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (200, -150)
    principled.inputs['Roughness'].default_value = 0.3
    principled.inputs['Metallic'].default_value = 0.2
    principled.inputs['Alpha'].default_value = alpha  # Set transparency

    # Emission shader for glow effect
    emission = nodes.new(type='ShaderNodeEmission')
    emission.location = (200, 100)
    emission.inputs['Strength'].default_value = emission_strength

    # Color Ramp for gradient
    color_ramp = nodes.new(type='ShaderNodeValToRGB')
    color_ramp.location = (-100, 0)
    # Set gradient colors (start to end)
    color_ramp.color_ramp.elements[0].color = start_color
    color_ramp.color_ramp.elements[1].color = end_color
    # Add middle color for more vibrant transition
    mid_elem = color_ramp.color_ramp.elements.new(0.5)
    mid_color = (
        (start_color[0] + end_color[0]) * 0.5 + 0.2,  # Slightly boosted
        (start_color[1] + end_color[1]) * 0.3,
        (start_color[2] + end_color[2]) * 0.6,
        1.0
    )
    mid_elem.color = mid_color

    # Spline Parameter node (gives 0-1 along the curve)
    # For curves, we can use the Generated texture coordinate's X component
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-500, 0)

    # Separate XYZ to get the spline parameter
    separate_xyz = nodes.new(type='ShaderNodeSeparateXYZ')
    separate_xyz.location = (-300, 0)

    # Connect nodes
    # Use Generated coordinates X for curve parameter (0-1 along spline)
    links.new(tex_coord.outputs['Generated'], separate_xyz.inputs['Vector'])
    links.new(separate_xyz.outputs['X'], color_ramp.inputs['Fac'])

    # Connect color to both emission and principled
    links.new(color_ramp.outputs['Color'], emission.inputs['Color'])
    links.new(color_ramp.outputs['Color'], principled.inputs['Base Color'])

    # Connect shaders to mix
    links.new(principled.outputs['BSDF'], mix_shader.inputs[1])
    links.new(emission.outputs['Emission'], mix_shader.inputs[2])

    # Connect to output
    links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])

    # Apply material to curve
    curve_obj.data.materials.append(mat)

    # Add arrowhead at the end of trajectory
    arrow_obj = None
    if add_arrow and len(positions) >= 2:
        # Calculate arrow direction (pointing in the direction of motion)
        arrow_dir = (positions[-1] - positions[-2]).normalized()
        arrow_pos = positions[-1]

        # Create a cone for the arrowhead
        arrow_height = line_thickness * 8 * arrow_scale
        arrow_radius = line_thickness * 2 * arrow_scale

        bpy.ops.mesh.primitive_cone_add(
            vertices=16,
            radius1=arrow_radius,
            radius2=0,
            depth=arrow_height,
            location=arrow_pos
        )
        arrow_obj = bpy.context.active_object
        arrow_obj.name = f"{curve_name}_Arrow"

        # Rotate arrow to point in the direction of motion
        # Default cone points in +Z, we need to align it with arrow_dir
        from mathutils import Matrix
        import math

        # Calculate rotation to align Z-axis with arrow_dir
        z_axis = Vector((0, 0, 1))
        rotation_axis = z_axis.cross(arrow_dir)
        if rotation_axis.length > 0.0001:
            rotation_axis.normalize()
            angle = math.acos(max(-1, min(1, z_axis.dot(arrow_dir))))
            rot_matrix = Matrix.Rotation(angle, 4, rotation_axis)
            arrow_obj.matrix_world = Matrix.Translation(arrow_pos) @ rot_matrix

            # Offset arrow so its base is at the curve end
            arrow_obj.location = arrow_pos + arrow_dir * (arrow_height * 0.5)
        else:
            # arrow_dir is parallel to Z axis
            if arrow_dir.z < 0:
                # Pointing down, flip
                arrow_obj.rotation_euler = (math.pi, 0, 0)
            arrow_obj.location = arrow_pos + arrow_dir * (arrow_height * 0.5)

        # Create solid color material for arrow (using end_color, no gradient)
        arrow_mat_name = f"{curve_name}_Arrow_Material"
        arrow_mat = bpy.data.materials.new(name=arrow_mat_name)
        arrow_mat.use_nodes = True

        # Enable transparency if alpha < 1
        if alpha < 1.0:
            arrow_mat.blend_method = 'BLEND'
            arrow_mat.shadow_method = 'HASHED'

        arrow_nodes = arrow_mat.node_tree.nodes
        arrow_links = arrow_mat.node_tree.links

        # Clear default nodes
        arrow_nodes.clear()

        # Output node
        arrow_output = arrow_nodes.new(type='ShaderNodeOutputMaterial')
        arrow_output.location = (600, 0)

        # Mix Shader to combine emission with principled for nice glow effect
        arrow_mix_shader = arrow_nodes.new(type='ShaderNodeMixShader')
        arrow_mix_shader.location = (400, 0)
        arrow_mix_shader.inputs['Fac'].default_value = 0.7  # More emission than diffuse

        # Principled BSDF for base color
        arrow_principled = arrow_nodes.new(type='ShaderNodeBsdfPrincipled')
        arrow_principled.location = (200, -150)
        arrow_principled.inputs['Roughness'].default_value = 0.3
        arrow_principled.inputs['Metallic'].default_value = 0.2
        arrow_principled.inputs['Alpha'].default_value = alpha  # Set transparency
        arrow_principled.inputs['Base Color'].default_value = end_color  # Use end_color as solid color

        # Emission shader for glow effect
        arrow_emission = arrow_nodes.new(type='ShaderNodeEmission')
        arrow_emission.location = (200, 100)
        arrow_emission.inputs['Strength'].default_value = emission_strength
        arrow_emission.inputs['Color'].default_value = end_color  # Use end_color as solid color

        # Connect shaders to mix
        arrow_links.new(arrow_principled.outputs['BSDF'], arrow_mix_shader.inputs[1])
        arrow_links.new(arrow_emission.outputs['Emission'], arrow_mix_shader.inputs[2])

        # Connect to output
        arrow_links.new(arrow_mix_shader.outputs['Shader'], arrow_output.inputs['Surface'])

        # Apply solid color material to arrow
        arrow_obj.data.materials.append(arrow_mat)
        print(f"Added arrowhead at end of trajectory '{curve_name}'")

    print(f"Root trajectory '{curve_name}' created with {len(positions)} points, alpha={alpha}, gradient from {start_color[:3]} to {end_color[:3]}")
    return curve_obj


def add_branch_trajectory(main_armature, virtual_armature, branch_color=(0.5, 0.9, 0.5, 1.0),
                          line_thickness=0.015, emission_strength=1.5, pelvis_height=1.0,
                          alpha=0.6, curve_name="BranchTrajectory"):
    """
    Add a trajectory line branching from main character to virtual character.
    This creates a visual connection showing the divergence point.

    Args:
        main_armature: Tuple of (armature, mesh_object_list) for main character
        virtual_armature: Tuple of (armature, mesh_object_list) for virtual character
        branch_color: RGBA color for the branch line
        line_thickness: Thickness of the trajectory line
        emission_strength: Emission strength for the glowing effect
        pelvis_height: Height of the pelvis/root joint from ground
        alpha: Transparency value (0=fully transparent, 1=opaque)
        curve_name: Name for the curve object

    Returns:
        The created curve object
    """
    from mathutils import Vector

    # Update scene to ensure poses are evaluated
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()

    positions = []

    # Get positions for both main and virtual armatures
    for armature, mesh_object_list in [main_armature, virtual_armature]:
        root_pos = None
        found_bone = False

        armature_eval = armature.evaluated_get(depsgraph)

        if armature_eval.pose and armature_eval.pose.bones:
            pelvis_names = ['pelvis', 'Pelvis', 'mixamorig:Hips', 'Hips', 'hips',
                           'm_avg_Pelvis', 'f_avg_Pelvis', 'root', 'Root']

            for bone_name in pelvis_names:
                if bone_name in armature_eval.pose.bones:
                    bone = armature_eval.pose.bones[bone_name]
                    bone_world_matrix = armature_eval.matrix_world @ bone.matrix
                    root_pos = bone_world_matrix.translation.copy()
                    found_bone = True
                    break

        if not found_bone:
            armature_loc = armature.location.copy()
            root_pos = Vector((armature_loc.x, armature_loc.y, armature_loc.z + pelvis_height))

        positions.append(root_pos)

    # Create a curve for the branch trajectory
    curve_data = bpy.data.curves.new(curve_name, type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 8
    curve_data.bevel_depth = line_thickness
    curve_data.bevel_resolution = 3
    curve_data.fill_mode = 'FULL'

    # Create a simple 2-point spline
    spline = curve_data.splines.new('BEZIER')
    spline.bezier_points.add(1)  # Add 1 more point (total 2)

    for i, pos in enumerate(positions):
        bp = spline.bezier_points[i]
        bp.co = pos
        bp.handle_left_type = 'AUTO'
        bp.handle_right_type = 'AUTO'

    # Create the curve object
    curve_obj = bpy.data.objects.new(curve_name, curve_data)
    bpy.context.collection.objects.link(curve_obj)

    # Create material with transparency
    mat_name = f"{curve_name}_Material"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True

    if alpha < 1.0:
        mat.blend_method = 'BLEND'
        mat.shadow_method = 'HASHED'

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)

    # Mix emission and principled for glow
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    mix_shader.location = (200, 0)
    mix_shader.inputs['Fac'].default_value = 0.6

    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (0, -100)
    principled.inputs['Base Color'].default_value = branch_color
    principled.inputs['Roughness'].default_value = 0.3
    principled.inputs['Alpha'].default_value = alpha

    emission = nodes.new(type='ShaderNodeEmission')
    emission.location = (0, 100)
    emission.inputs['Color'].default_value = branch_color
    emission.inputs['Strength'].default_value = emission_strength

    links.new(principled.outputs['BSDF'], mix_shader.inputs[1])
    links.new(emission.outputs['Emission'], mix_shader.inputs[2])
    links.new(mix_shader.outputs['Shader'], output.inputs['Surface'])

    curve_obj.data.materials.append(mat)

    return curve_obj



if __name__ == "__main__":
    # ${blender}  --python examples/render_smpl_seq_teaser.py -- /Users/shuaiqing/Documents/report/V2M_asserts/keling2_2b41fbd3.fbx
    parser = get_parser()
    args = parse_args(parser)

    fbxname = args.path

    setup()

    config_name = "keling2"
    virtual_name1 = "/Users/shuaiqing/Desktop/V2M/assets/h20_64_demo/epoch15/00000002_keling2_pred_seed42.fbx"
    virtual_name2 = "/Users/shuaiqing/Desktop/V2M/assets/h20_64_demo/epoch15/00000002_keling2_pred_seed43.fbx"
    default_trans = 0.7
    config = {
        "keling2": {
            # Main character frames
            "frames": [
                {"frame": 60, "x_offset": 0},
                {"frame": 80, "x_offset": 0},
                # {"frame": 85, "x_offset": 0},
                {"frame": 90, "x_offset": -0.15},
                {"frame": 95, "x_offset": 0.07},
                {"frame": 98, "x_offset": 0},
                # {"frame": 105, "x_offset": 0},
                {"frame": 110, "x_offset": 0},
                {"frame": 120, "x_offset": 0},
                {"frame": 130, "x_offset": 0},
                # {"frame": 150, "x_offset": 0},
                # {"frame": 180, "x_offset": 0},
            ],
            # Virtual characters config (dict format)
            # Each virtual trajectory follows main path until diverge_frame_idx, then branches
            "virtual": {
                "seed42": {
                    "filename": virtual_name1,
                    "y_offset": 1.5,
                    "diverge_frame_idx": 5,  # Diverge after frame index 5 (frame 98)
                    # Virtual frames starting from divergence point
                    "frames": [110, 120, 130],
                    "x_offset": 0.3,
                },
                "seed43": {
                    "filename": virtual_name2,
                    "y_offset": -1.5,
                    "diverge_frame_idx": 5,  # Diverge after frame index 5 (frame 98)
                    "frames": [110, 120, 130],
                    "x_offset": 0.3,
                },
            }
        }
    }[config_name]

    num_frames = len(config["frames"])

    # Collect armature list for trajectory visualization
    armature_list = []
    # Dict to store virtual armatures: {virtual_key: [(frame_idx, armature, mesh_list), ...]}
    virtual_armature_dict = {}

    # Step 1: Load main character frames
    for ii, frame_config in enumerate(config["frames"]):
        color_progress = ii / max(num_frames - 1, 1)
        x_offset = frame_config["x_offset"] + default_trans * ii
        print(f"Loading main frame {frame_config['frame']} at x={x_offset:.2f}, color_progress={color_progress:.2f}")
        armature, mesh_object_list = load_fbx_at_frame(
            fbxname,
            frame_config["frame"],
            x_offset,
            instance_id=ii,
            color_progress=color_progress,
            target_frame=1
        )
        armature_list.append((armature, mesh_object_list))

    # Step 2: Load virtual characters from dict config
    if "virtual" in config:
        for vidx, (vkey, vconfig) in enumerate(config["virtual"].items()):
            virtual_filename = vconfig["filename"]
            y_offset = vconfig["y_offset"]
            diverge_idx = vconfig["diverge_frame_idx"]
            virtual_frames = vconfig["frames"]

            virtual_armature_dict[vkey] = {
                "diverge_idx": diverge_idx,
                "armatures": []
            }

            for vf_idx, vframe in enumerate(virtual_frames):
                # Find the corresponding main frame index for x_offset calculation
                # Virtual frames start after diverge_idx
                main_frame_idx = diverge_idx + vf_idx
                if main_frame_idx < len(config["frames"]):
                    x_offset = config["frames"][main_frame_idx]["x_offset"] + default_trans * main_frame_idx
                else:
                    x_offset = default_trans * main_frame_idx
                x_offset += vconfig["x_offset"]

                color_progress = main_frame_idx / max(num_frames - 1, 1)
                print(f"  Loading virtual '{vkey}' frame {vframe} at x={x_offset:.2f}, y={y_offset}")

                v_armature, v_mesh_list = load_fbx_at_frame(
                    virtual_filename,
                    vframe,
                    x_offset,
                    instance_id=f"{main_frame_idx}_v{vidx}",
                    color_progress=color_progress,
                    y_offset=y_offset,
                    is_virtual=True,
                    virtual_alpha=0.55,
                )
                virtual_armature_dict[vkey]["armatures"].append((v_armature, v_mesh_list))

    center = default_trans * (num_frames - 1) / 2

    # Add root trajectory line for main character
    # Extended at both ends to "pierce through" the body, with arrow at the end
    trajectory_curve = add_root_trajectory(
        armature_list,
        start_color=(0.2, 0.7, 1.0, 1.0),   # Cyan blue (start of motion)
        end_color=(1.0, 0.2, 0.6, 1.0),     # Magenta pink (end of motion)
        line_thickness=0.025,
        emission_strength=2.5,
        curve_name="MainTrajectory",
        extend_start=0.8,    # Extend before first character
        extend_end=0.8,      # Extend after last character
        add_arrow=True,      # Add arrowhead at the end
        arrow_scale=1.2      # Slightly larger arrow
    )

    # Create virtual trajectories that branch from main trajectory
    # Each virtual trajectory: main[0:diverge_idx+1] + virtual[diverge_idx+1:]
    virtual_colors = [
        ((0.4, 0.9, 0.5, 1.0), (0.2, 0.7, 0.3, 1.0)),   # Green gradient
        ((0.9, 0.7, 0.3, 1.0), (0.8, 0.4, 0.2, 1.0)),   # Orange gradient
    ]

    for vidx, (vkey, vdata) in enumerate(virtual_armature_dict.items()):
        diverge_idx = vdata["diverge_idx"]
        v_armatures = vdata["armatures"]

        # Build combined trajectory: main path up to diverge point, then virtual path
        # Take main armatures from 0 to diverge_idx (inclusive)
        combined_armature_list = armature_list[:diverge_idx] + v_armatures

        if len(combined_armature_list) >= 2:
            color_pair = virtual_colors[vidx % len(virtual_colors)]
            v_trajectory = add_root_trajectory(
                combined_armature_list,
                start_color=color_pair[0],
                end_color=color_pair[1],
                line_thickness=0.018,
                emission_strength=1.8,
                alpha=0.65,
                curve_name=f"VirtualTrajectory_{vkey}",
                extend_start=0.0,    # No extension at start (branches from main)
                extend_end=0.6,      # Extend after last virtual character
                add_arrow=True,      # Add arrowhead at the end
                arrow_scale=0.9      # Slightly smaller arrow for virtual
            )
            print(f"Created virtual trajectory '{vkey}': main[0:{diverge_idx+1}] + virtual[{len(v_armatures)} frames]")

    # Create L-shaped seamless studio backdrop (cyclorama)
    # This replaces the flat checkerboard ground
    # backdrop = create_studio_backdrop(
    #     center=(center, 0, 0),
    #     width=25,           # Wide enough to cover the scene
    #     depth=12,           # Depth from front to back wall
    #     height=8,           # Height of back wall
    #     bevel_radius=2.5,   # Smooth curved transition
    #     color=(0.98, 0.98, 0.98, 1.0)  # Near-white studio backdrop for bright look
    # )

    # Also create a reflective floor on top of the backdrop for subtle reflections
    ground = build_plane(translation=(center, 0, 0.001), plane_size=100)  # Slightly above backdrop
    # add_reflection_to_ground(
    #     ground,
    #     roughness=0.08,   # Low roughness for ceramic tile glossy look
    #     metallic=0.0,     # Non-metallic for ceramic
    #     specular=0.9,     # High specular for ceramic tile reflections
    #     white_color=(1.0, 1.0, 1.0, 1.0),    # Pure bright white
    #     # gray_color=(70/255, 90/255, 100/255, 1.0)   # Very light gray, almost white
    #     gray_color=(70/255, 200/255, 70/255, 1.0)   # 正绿色
    # )

    # Camera setup: 85mm focal length, lower angle for more level view
    # 85mm is a portrait lens that compresses perspective nicely
    # Lower camera position and look at chest height for level feel
    camera_height = 15  # Lower camera, closer to eye level
    camera_distance = 17.0  # Adjusted for 85mm focal length
    look_at_height = 1.0  # Look at chest/torso level

    set_camera(
        location=(center, camera_distance, camera_height),
        center=(center, 0, look_at_height),
        focal=85  # Portrait lens focal length
    )

    # Setup professional balanced lighting from both sides
    # Even illumination to avoid left-dark / right-bright issue
    # All lights now use track_to constraint to aim at subject center
    setup_studio_three_point_lighting(
        center=(center, 0, 1.0),  # Target at human torso height
        key_strength=300.0  # Increased strength for well-lit subjects
    )

    # Add subtle fog effect for depth
    # Camera is at Y=camera_distance, subjects are at Y=0
    # So subjects are ~camera_distance units away from camera
    # Fog should start slightly before subjects and gradually increase behind them
    fog_start = 17.0  # 从距离相机 2米处就开始起雾（这就很浓了）
    fog_depth = 25.0 # 雾在 25米 范围内渐变为全白

    # setup_mist_fog(
    #     bpy.context.scene,
    #     start=fog_start,
    #     depth=fog_depth,
    #     fog_color=(0.8, 0.85, 0.95) # 稍微蓝一点，证明雾存在
    # )
    # print(f"Fog configured: start={fog_start}, depth={fog_depth}, camera_distance={camera_distance}")
    if args.debug:
        set_cycles_renderer(
            bpy.context.scene,
            bpy.data.objects["Camera"],
            num_samples=16,
            use_transparent_bg=False,
            use_denoising=True,
        )
    else:
        set_cycles_renderer(
            bpy.context.scene,
            bpy.data.objects["Camera"],
            num_samples=512,
            use_transparent_bg=False,
            use_denoising=True,
        )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f'output/v2m_teaser_{timestamp}.png'

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    set_output_properties(
        bpy.context.scene,
        output_file_path=output_path,
        res_x=(2048+1024)//2,
        res_y=(1024+1024)//2,
        tile_x=512,
        tile_y=512,
        resolution_percentage=100,
        format='JPEG'
    )

    # write_still=True is required to save the rendered image to file
    if not args.debug:
        render_with_progress(write_still=True)