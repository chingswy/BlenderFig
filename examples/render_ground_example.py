import bpy
from myblender.geometry import build_plane
from myblender.setup import setup, set_output_properties
from myblender.setup import set_cycles_renderer, add_sunlight
from myblender.geometry import set_camera



if __name__ == "__main__":
    # ${blender} --python examples/render_ground_example.py
    setup(rgb=(1,1,1,0))
    build_plane(translation=(0, 0, 0), plane_size=20, white=(1,1,1,1), black=(0.1,0.8,0.1,1))

    add_sunlight(
        location=(5, 5, 5),
        lookat=(0, 0, 1),
    )

    set_camera(
        location=(0, 10, 2),
        center=(0, 0, 1),
        focal=85  # Portrait lens focal length
    )

    set_output_properties(
        bpy.context.scene,
        output_file_path="examples/render_ground_example.png",
        res_x=2048,
        res_y=2048,
        tile_x=2048,
        tile_y=2048,
    )

    set_cycles_renderer(
        bpy.context.scene,
        bpy.data.objects["Camera"],
        num_samples=32,
        use_transparent_bg=False,
        use_denoising=True,
    )