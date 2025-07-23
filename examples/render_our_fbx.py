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
)
from myblender.material import colorObj, setMat_plastic


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

if __name__ == '__main__':
    # ${blender} --background -noaudio --python examples/render_our_fbx.py -- ~/Desktop/t2m/swimanimset_jog_fwd_in_shallow_water.fbx
    parser = get_parser()
    parser.add_argument('--path', type=str, default=None)
    args = parse_args(parser)

    setup()
    add_sunlight(name='Light', location=(0., 0., 5.), rotation=(0., np.pi/12, 0), strength=2.0)
    

    set_world_background()
    # setLight_ambient(color=(0.1,0.1,0.1,1))
    scene = set_eevee_renderer()
    # set_cycles_renderer(
    #     bpy.context.scene,
    #     bpy.data.objects["Camera"],
    #     num_samples=64,
    #     use_transparent_bg=False,
    #     use_denoising=True,
    # )

    build_plane(translation=(0, 0, 0), plane_size=20)

    fbx_path = args.path
    assert os.path.exists(fbx_path), fbx_path
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    # Get the imported objects
    obj_names = [o.name for o in bpy.context.selected_objects]
    
    # Find the armature (assuming it's the first object or has animation data)
    armature = None
    mesh_object = None
    for obj_name in obj_names:
        obj = bpy.data.objects[obj_name]
        print(obj_name, obj.type)
        if obj.type == 'ARMATURE' or (obj.animation_data and obj.animation_data.action):
            armature = obj
        if obj.type == 'MESH':
            mesh_object = obj
    
    set_scene_frame_range(armature)
    base_location = (0, -3, 0.5)
    # set_camera(location=(0, -4, 2.), center=(0, 0, 1), focal=30)

    world_bound_box = [mesh_object.matrix_world @ Vector(corner) for corner in mesh_object.bound_box]
    min_x, min_y, min_z = world_bound_box[0]
    max_x, max_y, max_z = world_bound_box[6]
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2
    print(center_x, center_y, center_z)
    set_camera(
        location=(base_location[0] + center_x, base_location[1] + center_y, base_location[2] + center_z), 
        center=(center_x, center_y, center_z), focal=30)
    # Add keyframe for camera position and rotation at the first frame
    bpy.context.scene.frame_set(bpy.context.scene.frame_start)
    camera = bpy.data.objects['Camera']
    camera.keyframe_insert(data_path="location", frame=bpy.context.scene.frame_start)
    camera.keyframe_insert(data_path="rotation_euler", frame=bpy.context.scene.frame_start)
    
    print(f"Added keyframe at frame {bpy.context.scene.frame_start} for camera position and rotation")
    # Switch to the last frame to calculate center at the end of animation
    if armature and armature.animation_data and armature.animation_data.action:
        # Get the last frame from the animation
        last_frame = bpy.context.scene.frame_end
        
        # Set the current frame to the last frame
        bpy.context.scene.frame_set(last_frame)
        
        
        # Recalculate the bounding box and center at the last frame
        world_bound_box_last = [mesh_object.matrix_world @ Vector(corner) for corner in mesh_object.bound_box]
        min_x_last, min_y_last, min_z_last = world_bound_box_last[0]
        max_x_last, max_y_last, max_z_last = world_bound_box_last[6]
        center_x_last = (min_x_last + max_x_last) / 2
        center_y_last = (min_y_last + max_y_last) / 2
        center_z_last = (min_z_last + max_z_last) / 2
        
        print(f"Last frame center: {center_x_last}, {center_y_last}, {center_z_last}")
        
        # Update camera to look at the last frame position
        set_camera(
            location=(base_location[0] + center_x_last, base_location[1] + center_y_last, base_location[2] + center_z_last), 
            center=(center_x_last, center_y_last, center_z_last), focal=30)
        # Add keyframe for camera position and rotation at the last frame
        camera.keyframe_insert(data_path="location", frame=last_frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=last_frame)
    
        print(f"Added keyframe at frame {last_frame} for camera position and rotation")
    
    # Set interpolation type to make camera movement smoother
    # for fcurve in camera.animation_data.action.fcurves:
    #     for kf in fcurve.keyframe_points:
    #         kf.interpolation = 'BEZIER'

    # Apply material to the mesh object, not the armature
    if mesh_object:
        meshColor = colorObj((153/255.,  216/255.,  201/255., 1.), 0.5, 1.0, 1.0, 0.0, 2.0)
        setMat_plastic(mesh_object, meshColor, roughness=0.9, metallic=0.5, specular=0.5)
    

    set_output_properties(bpy.context.scene, output_file_path='./output_eevee/', 
        res_x=1024, res_y=1024, 
        tile_x=1024, tile_y=1024, 
        resolution_percentage=100,
        format='JPEG'
    )
    bpy.ops.render.render(animation=True)