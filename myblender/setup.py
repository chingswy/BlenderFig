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

def add_sunlight(name='Light', location=(10., 0., 5.), rotation=(0., -np.pi/4, 3.14), strength=4.):
    bpy.ops.object.light_add(type='SUN', location=location, rotation=rotation)

    if name is not None:
        bpy.context.object.name = name

    sun_object = bpy.context.object
    sun_object.data.use_nodes = True
    sun_object.data.node_tree.nodes["Emission"].inputs["Strength"].default_value = strength

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
                        prefer_cuda_use: bool = True,
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
    if prefer_cuda_use:
        bpy.context.scene.cycles.device = "GPU"

        # Change the preference setting
        try:
            bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        except:
            print("No CUDA device found, using CPU instead.")

    # Call get_devices() to let Blender detects GPU device (if any)
    bpy.context.preferences.addons["cycles"].preferences.get_devices()

    # Let Blender use all available devices, include GPU and CPU
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        d["use"] = 1

    # Display the devices to be used for rendering
    print("----")
    print("The following devices will be used for path tracing:")
    for d in bpy.context.preferences.addons["cycles"].preferences.devices:
        print("- {}".format(d["name"]))
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