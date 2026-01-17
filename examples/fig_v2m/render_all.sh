export PYTHONPATH=.
export blender=/Applications/Blender3.app/Contents/MacOS/Blender

# 渲染序列
body_color="0.05 0.326 1."
ground_color="0.78 0.78 0.78 1."
${blender} --background --python examples/fig_v2m/render_ground_seq.py -- --body ${body_color} --ground ${ground_color}
# # 渲染两个视角
# ${blender} --background --python examples/fig_v2m/render_ground_fbx.py -- --body ${body_color} --ground ${ground_color} --frame 60
# ${blender} --background --python examples/fig_v2m/render_ground_fbx.py -- --body ${body_color} --ground ${ground_color} --frame 90
# ${blender} --background --python examples/fig_v2m/render_ground_fbx.py -- --body ${body_color} --ground ${ground_color} --frame 95

# # 渲染多样性
# ${blender} --background --python examples/fig_v2m/render_diverse.py -- --body ${body_color} --frame 110
# ${blender} --background --python examples/fig_v2m/render_diverse.py -- --body ${body_color} --frame 130


# ${blender} --background --python examples/fig_v2m/render_diverse.py -- --body ${body_color} --frame 22
# ${blender} --background --python examples/fig_v2m/render_diverse.py -- --body ${body_color} --frame 62

