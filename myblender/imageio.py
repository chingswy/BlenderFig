import bpy
import os

def imwrite(outname, rgb):
    os.makedirs(os.path.dirname(outname), exist_ok=True)
    size = rgb.shape
    image = bpy.data.images.new("MyImage", width=size[0], height=size[1])
    # assign pixels
    image.pixels = rgb.flatten()

    # write image
    image.filepath_raw = outname
    image.file_format = 'JPEG'
    image.save()