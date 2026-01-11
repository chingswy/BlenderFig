import os
import glob

def render_smpl_shot_start(blender, root):
    for case in sorted(os.listdir(root)):
        for fbxname in glob.glob(os.path.join(root, case, '*.fbx')):
            outname = fbxname.replace('.fbx', '.png')
            if os.path.exists(outname):
                continue
            cmd = f'{blender} --background -noaudio --python examples/render_smpl_shot.py -- {fbxname} --num_frames 6 --skip_start 10 --skip_end 10 --out {outname}'
            os.system(cmd)

if __name__ == '__main__':
    blender = '/Applications/Blender.app/Contents/MacOS/Blender'

    for root in [
        # '/Users/shuaiqing/Documents/report/report_comparison_3/ours_better_all',
        # '/Users/shuaiqing/Documents/report/report_comparison_3/ours_better_avg',
        '/Users/shuaiqing/Documents/report/report_comparison_3/ours_better_full',
    ]:
        render_smpl_shot_start(blender, root)