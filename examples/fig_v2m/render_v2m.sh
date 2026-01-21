# 这个脚本用来完整产出所有的blender渲染的数据
export blender=/apdcephfs_cq11/share_1467498/softwares/blender-3.6.18-linux-x64/blender
export PYTHONPATH=.
${blender} --background --python examples/fig_v2m/render_fbx_video3d.py -- --name baichuyu_wovideo --num_samples 256

${blender} --background --python examples/fig_v2m/render_fbx_video3d.py -- --name FWUCjj44YIg_71 --num_samples 512

${blender} --background --python examples/fig_v2m/render_fbx_video3d.py -- --name FWUCjj44YIg_71_over --num_samples 512


