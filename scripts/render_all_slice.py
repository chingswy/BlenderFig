import os

if __name__ == '__main__':
    blender = '/dellnas/home/yuzhiyuan/blender/blender-2.93.15-linux-x64/blender'

    for offset in range(-20, 20):
        _offset = offset / 100
        out = f'output/skelfield/{offset}'
        cmd0 = f'python3 scripts/generate_skelfield.py assets/s04_Hug1_000085.jpg.json {out} --offset {_offset}'
        cmd1 = f'{blender} -noaudio --background --python examples/render_skelfield.py -- {out}_records.json --skel panoptic15 --out {out}_output.jpg'
        os.sysmte(cmd0)
        os.sysmte(cmd1)