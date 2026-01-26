'''
  @ Date: 2022-04-24 15:21:36
  @ Author: Qing Shuai
  @ Mail: s_q@zju.edu.cn
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2022-09-05 19:51:27
  @ FilePath: /EasyMocapPublic/easymocap/blender/setup.py
'''
import bpy
import numpy as np
import sys
from mathutils import Matrix, Vector, Quaternion, Euler

def get_parser():
    # Args
    import argparse
    parser = argparse.ArgumentParser(
        usage='''
    render: `blender --background -noaudio --python ./scripts/blender/render_camera.py -- ${data} --nf 90`
''',
        description='render example')
    parser.add_argument('path', type=str,
        help='Input file or directory')
    parser.add_argument("--nf", type=int, help='frame', default=1)
    parser.add_argument("--nperson", type=int, default=6)
    parser.add_argument("--extra_mesh", type=str, default=[], nargs='+')
    parser.add_argument('--out', type=str, default=None,
        help='Output file or directory')
    parser.add_argument('--out_blend', type=str, default=None,
        help='Output file or directory')
    parser.add_argument('--tmp', type=str, default='/dockerdata/render',
        help='Temporary directory for video output during rendering')
    parser.add_argument('--res_x', type=int, default=1024)
    parser.add_argument('--res_y', type=int, default=1024)
    parser.add_argument('--format', type=str, default='JPEG', choices=['JPEG', 'PNG'])

    parser.add_argument('--res', type=int, default=100,
        help='Output file or directory')
    parser.add_argument('--num_samples', type=int, default=128,
        help='Output file or directory')
    parser.add_argument('--denoising', action='store_true')
    parser.add_argument('--nocycle', action='store_true')
    parser.add_argument('--debug', action='store_true')
    return parser

def parse_args(parser):
    if '--' in sys.argv:
        args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    else:
        args = parser.parse_args(['debug'])
    return args

def clean_objects(name='Cube', version = '2.83') -> None:
    if name not in bpy.data.objects.keys():
        return 0
    bpy.ops.object.select_all(action='DESELECT')
    if version == '2.83':
        bpy.data.objects[name].select_set(True)
    else:
        bpy.data.objects[name].select = True
    bpy.ops.object.delete(use_global=False)

def add_sunlight(name='Light', location=(10., 0., 5.), rotation=(0., -np.pi/4, 3.14),
                 lookat=None, strength=4., cast_shadow=True):
    """Add a sun light to the scene.

    Args:
        name: Name of the light object
        location: Position of the light (x, y, z)
        rotation: Euler rotation angles (rx, ry, rz) in radians. Ignored if lookat is provided.
        lookat: Target point (x, y, z) for the light to look at. If provided, overrides rotation.
        strength: Light emission strength

    Returns:
        The created sun light object
    """
    bpy.ops.object.light_add(type='SUN', location=location)

    if name is not None:
        bpy.context.object.name = name

    sun_object = bpy.context.object

    # Set rotation: use lookat if provided, otherwise use rotation
    if lookat is not None:
        lookat = np.array(lookat)
        loc = np.array(location)
        direction = Vector(lookat - loc)
        # Point the light's '-Z' axis toward the target and use 'Y' as up
        rot_quat = direction.to_track_quat('-Z', 'Y')
        sun_object.rotation_euler = rot_quat.to_euler()
    else:
        sun_object.rotation_euler = rotation

    sun_object.data.use_nodes = True
    sun_object.data.node_tree.nodes["Emission"].inputs["Strength"].default_value = strength
    # Set shadow casting (for both Eevee and Cycles)
    sun_object.data.use_shadow = cast_shadow
    # Cycles-specific shadow setting
    if hasattr(sun_object.data, 'cycles'):
        sun_object.data.cycles.cast_shadow = cast_shadow
    
    return sun_object

def add_area_light(name='Light', location=(10., 0., 5.), rotation=(0., -np.pi/4, 3.14),
                 lookat=None, strength=4., size=1.0, size_y=None, shape='SQUARE'):
    bpy.ops.object.light_add(type='AREA', location=location)
    area_object = bpy.context.object
    area_object.name = name
    area_object.data.use_nodes = True
    area_object.data.node_tree.nodes["Emission"].inputs["Strength"].default_value = strength
    
    # 设置面光源形状和尺寸
    area_object.data.shape = shape
    area_object.data.size = size
    if size_y is not None and shape in ('RECTANGLE', 'ELLIPSE'):
        area_object.data.size_y = size_y
    
    if lookat is not None:
        lookat = np.array(lookat)
        loc = np.array(location)
        direction = Vector(lookat - loc)
        rot_quat = direction.to_track_quat('-Z', 'Y')
        area_object.rotation_euler = rot_quat.to_euler()
    return area_object

def add_spot_light(name='SpotLight', location=(10., 0., 5.), rotation=(0., -np.pi/4, 3.14),
                   lookat=None, strength=100., spot_size=np.pi/4, spot_blend=0.15,
                   shadow_soft_size=0.25, cast_shadow=True):
    """Add a spot light to the scene.

    Args:
        name: Name of the light object
        location: Position of the light (x, y, z)
        rotation: Euler rotation angles (rx, ry, rz) in radians. Ignored if lookat is provided.
        lookat: Target point (x, y, z) for the light to look at. If provided, overrides rotation.
        strength: Light emission strength (default 100, spot lights typically need higher values)
        spot_size: Cone angle in radians (default pi/4 = 45 degrees)
        spot_blend: Edge softness, 0-1 (default 0.15)
        shadow_soft_size: Soft shadow radius (default 0.25)
        cast_shadow: Whether the light casts shadows (default True)

    Returns:
        The created spot light object
    """
    bpy.ops.object.light_add(type='SPOT', location=location)
    spot_object = bpy.context.object
    
    if name is not None:
        spot_object.name = name
    
    # Set rotation: use lookat if provided, otherwise use rotation
    if lookat is not None:
        lookat = np.array(lookat)
        loc = np.array(location)
        direction = Vector(lookat - loc)
        rot_quat = direction.to_track_quat('-Z', 'Y')
        spot_object.rotation_euler = rot_quat.to_euler()
    else:
        spot_object.rotation_euler = rotation
    
    # Set spot light specific properties
    spot_object.data.spot_size = spot_size
    spot_object.data.spot_blend = spot_blend
    spot_object.data.shadow_soft_size = shadow_soft_size
    
    # Set shadow casting (for both Eevee and Cycles)
    spot_object.data.use_shadow = cast_shadow
    # Cycles-specific shadow setting
    if hasattr(spot_object.data, 'cycles'):
        spot_object.data.cycles.cast_shadow = cast_shadow
    
    # Set emission strength using nodes
    spot_object.data.use_nodes = True
    spot_object.data.node_tree.nodes["Emission"].inputs["Strength"].default_value = strength
    
    return spot_object


def setLight_sun(rotation_euler, strength, shadow_soft_size = 0.05):
	x = rotation_euler[0] * 1.0 / 180.0 * np.pi
	y = rotation_euler[1] * 1.0 / 180.0 * np.pi
	z = rotation_euler[2] * 1.0 / 180.0 * np.pi
	angle = (x,y,z)
	bpy.ops.object.light_add(type = 'SUN', rotation = angle)
	lamp = bpy.data.lights['Sun']
	lamp.use_nodes = True
	# lamp.shadow_soft_size = shadow_soft_size # this is for older blender 2.8
	lamp.angle = shadow_soft_size

	lamp.node_tree.nodes["Emission"].inputs['Strength'].default_value = strength
	return lamp

def setLight_ambient(color = (0,0,0,1)):
	bpy.data.scenes[0].world.use_nodes = True
	bpy.data.scenes[0].world.node_tree.nodes["Background"].inputs['Color'].default_value = color


def setup(rgb=(1,1,1,1)):
    np.random.seed(666)
    scene = bpy.context.scene
    build_rgb_background(scene.world, rgb=rgb, strength=1.)
    clean_objects('Cube')
    clean_objects('Light')

def build_rgb_background(world,
                         rgb = (0.9, 0.9, 0.9, 1.0),
                         strength = 1.0) -> None:
    world.use_nodes = True
    node_tree = world.node_tree

    rgb_node = node_tree.nodes.new(type="ShaderNodeRGB")
    rgb_node.outputs["Color"].default_value = rgb

    node_tree.nodes["Background"].inputs["Strength"].default_value = strength

    node_tree.links.new(rgb_node.outputs["Color"], node_tree.nodes["Background"].inputs["Color"])

def set_eevee_renderer(scene: bpy.types.Scene,
    camera_object: bpy.types.Object):
    scene.camera = camera_object

    # Set up Eevee renderer
    # In newer Blender versions, 'BLENDER_EEVEE' is replaced with 'BLENDER_EEVEE_NEXT'
    try:
        scene.render.engine = 'BLENDER_EEVEE'
    except TypeError:
        # Fall back to BLENDER_EEVEE_NEXT if BLENDER_EEVEE is not available
        scene.render.engine = 'BLENDER_EEVEE_NEXT'

    # Configure Eevee settings for faster rendering
    scene.eevee.taa_render_samples = 16  # Reduce samples for faster rendering
    # scene.eevee.use_soft_shadows = True
    # Disable screen space reflections for speed
    try:
        scene.eevee.use_ssr = False
    except AttributeError:
        # Handle newer Blender versions where the attribute might have a different name
        if hasattr(scene.eevee, 'ssr_enable'):
            scene.eevee.ssr_enable = False
    # Handle SSR refraction setting for compatibility across Blender versions
    try:
        scene.eevee.use_ssr_refraction = False
    except AttributeError:
        # For newer Blender versions where the attribute might have a different name
        if hasattr(scene.eevee, 'ssr_refraction_enable'):
            scene.eevee.ssr_refraction_enable = False
    scene.eevee.use_gtao = True  # Keep ambient occlusion for better visuals
    scene.eevee.gtao_distance = 0.2
    # Handle bloom setting for compatibility across Blender versions
    try:
        scene.eevee.use_bloom = False  # Disable bloom for speed
    except AttributeError:
        # For newer Blender versions where the attribute might have a different name
        if hasattr(scene.eevee, 'bloom_enable'):
            scene.eevee.bloom_enable = False

def set_cycles_renderer(scene: bpy.types.Scene,
                        camera_object: bpy.types.Object,
                        num_samples: int,
                        use_denoising: bool = True,
                        use_motion_blur: bool = False,
                        use_transparent_bg: bool = True,
                        prefer_gpu: bool = True,
                        use_adaptive_sampling: bool = False) -> None:
    scene.camera = camera_object

    scene.render.engine = 'CYCLES'
    scene.render.use_motion_blur = use_motion_blur

    scene.render.film_transparent = use_transparent_bg
    scene.view_layers[0].cycles.use_denoising = use_denoising

    scene.cycles.use_adaptive_sampling = use_adaptive_sampling
    scene.cycles.samples = num_samples

    # Enable GPU acceleration
    # Source - https://blender.stackexchange.com/a/196702
    if prefer_gpu:
        bpy.context.scene.cycles.device = "GPU"

        # Try different compute device types in order of preference
        # METAL for Mac, CUDA for NVIDIA, OPTIX for newer NVIDIA, HIP for AMD
        import sys
        compute_device_types = []
        if sys.platform == "darwin":
            # Mac uses Metal
            compute_device_types = ["METAL", "NONE"]
        else:
            # Windows/Linux: try CUDA, OPTIX, HIP
            compute_device_types = ["CUDA", "OPTIX", "HIP", "NONE"]

        cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
        device_set = False

        for device_type in compute_device_types:
            try:
                cycles_prefs.compute_device_type = device_type
                cycles_prefs.get_devices()
                # Check if we have any GPU devices available
                gpu_devices = [d for d in cycles_prefs.devices if d.type != 'CPU']
                if gpu_devices or device_type == "NONE":
                    print(f"Using compute device type: {device_type}")
                    device_set = True
                    break
            except Exception as e:
                print(f"Could not use {device_type}: {e}")
                continue

        if not device_set:
            print("No GPU device found, using CPU instead.")
            bpy.context.scene.cycles.device = "CPU"

    # Call get_devices() to let Blender detects GPU device (if any)
    bpy.context.preferences.addons["cycles"].preferences.get_devices()

    # Let Blender use all available devices, include GPU and CPU
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1

    # Display the devices to be used for rendering
    print("----")
    print("The following devices will be used for path tracing:")
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        print("- {} (type: {}, use: {})".format(d["name"], d["type"], d["use"]))
    print("----")

def set_output_properties(scene,
                          resolution_percentage: int = 100,
                          output_file_path: str = "",
                          res_x: int = 1920,
                          res_y: int = 1080,
                          tile_x: int = 1920,
                          tile_y: int = 1080,
                          format='PNG') -> None:
    scene.render.resolution_percentage = resolution_percentage
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    if hasattr(scene.render, 'tile_x'):
        scene.render.tile_x = tile_x
        scene.render.tile_y = tile_y
    # scene.render.use_antialiasing = True
    # scene.render.antialiasing_samples = '5'

    scene.render.filepath = output_file_path
    if format == 'PNG':
        scene.render.image_settings.file_format = "PNG"
        # scene.render.alpha_mode = "TRANSPARENT"
        scene.render.image_settings.color_mode = "RGBA"
    elif format == 'JPEG':
        scene.render.image_settings.file_format = "JPEG"
        scene.render.image_settings.color_mode = "RGB"
    elif format == 'FFMPEG':
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.image_settings.color_mode = "RGB"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        # scene.render.ffmpeg.quality = 90
    else:
        raise ValueError(f"Unsupported format: {format}")


def render_with_progress(write_still: bool = True) -> None:
    """
    Render the current scene with progress display in terminal.

    Args:
        write_still: Whether to save the rendered image to file
    """
    import time

    scene = bpy.context.scene
    total_samples = scene.cycles.samples

    # Progress tracking variables
    render_start_time = None
    last_progress = -1

    def progress_callback(dummy):
        nonlocal render_start_time, last_progress

        if render_start_time is None:
            render_start_time = time.time()

        # Get current progress (0.0 to 1.0)
        # In Cycles, we can estimate progress from render result
        progress = bpy.context.scene.render.progress if hasattr(bpy.context.scene.render, 'progress') else 0

        # Calculate elapsed time
        elapsed = time.time() - render_start_time

        # Create progress bar
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        percent = progress * 100

        # Estimate remaining time
        if progress > 0.01:
            eta = elapsed / progress * (1 - progress)
            eta_str = f"ETA: {eta:.0f}s"
        else:
            eta_str = "ETA: --"

        # Only print if progress changed significantly
        current_percent = int(percent)
        if current_percent != last_progress:
            last_progress = current_percent
            print(f"\r渲染进度: |{bar}| {percent:5.1f}% | 耗时: {elapsed:.0f}s | {eta_str}    ", end='', flush=True)

    # Alternative: Use render handlers for progress
    # Register a timer that checks render progress
    print(f"\n开始渲染... (采样数: {total_samples})")
    print(f"输出路径: {scene.render.filepath}")
    print("-" * 60)

    # For Cycles, enable progress reporting
    if hasattr(scene.cycles, 'use_progressive_refine'):
        scene.cycles.use_progressive_refine = True

    # Use a simpler approach: render with write_still and track via handler
    render_start = time.time()

    # Define handler functions
    def render_init(scene):
        nonlocal render_start
        render_start = time.time()
        print(f"渲染初始化完成...")

    def render_pre(scene):
        print(f"开始渲染帧...")

    def render_post(scene):
        elapsed = time.time() - render_start
        print(f"\n渲染完成! 总耗时: {elapsed:.1f}秒")

    def render_stats(scene):
        # This is called periodically during render
        elapsed = time.time() - render_start
        # Try to get sample info from stats
        if hasattr(bpy.context, 'view_layer') and bpy.context.view_layer:
            stats = scene.statistics(bpy.context.view_layer)
            print(f"\r{stats} | 耗时: {elapsed:.0f}s    ", end='', flush=True)

    # Register handlers
    bpy.app.handlers.render_init.append(render_init)
    bpy.app.handlers.render_pre.append(render_pre)
    bpy.app.handlers.render_post.append(render_post)

    try:
        # Perform the render
        bpy.ops.render.render(animation=False, write_still=write_still)

        elapsed = time.time() - render_start
        print(f"\n" + "=" * 60)
        print(f"✓ 渲染完成!")
        print(f"  采样数: {total_samples}")
        print(f"  总耗时: {elapsed:.1f}秒")
        print(f"  输出: {scene.render.filepath}")
        print("=" * 60)

    finally:
        # Cleanup handlers
        if render_init in bpy.app.handlers.render_init:
            bpy.app.handlers.render_init.remove(render_init)
        if render_pre in bpy.app.handlers.render_pre:
            bpy.app.handlers.render_pre.remove(render_pre)
        if render_post in bpy.app.handlers.render_post:
            bpy.app.handlers.render_post.remove(render_post)