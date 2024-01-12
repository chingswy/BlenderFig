# Blender

## install

```bash
<path_to_blender>/2.93/python/bin/python3.9 -m ensurepip --upgrade
<path_to_blender>/2.93/python/bin/python3.9 setup.py develop
# For example, in my MacOS:
/Applications/Blender.app/Contents/Resources/2.82/python/bin/python3.7m -m ensurepip --upgrade
/Applications/Blender.app/Contents/Resources/2.82/python/bin/python3.7m setup.py develop
/Applications/Blender.app/Contents/Resources/2.82/python/bin/python3.7m install PyYaml ipdb
```

## render_example.py

In this example, we show how to render some simple primitives:

```bash
${blender} --background -noaudio --python examples/render_example.py -- xxx --out output/render_example.jpg --out_blend output/render_example.blend
```

![](output/render_example.jpg)

## render grid

```bash
${blender} --background -noaudio --python examples/render_grid.py -- debug --out output/render_grid.jpg --out_blend output/render_grid.blend
```

![](output/render_grid.jpg)

## render multiple images

```bash
${blender} --background -noaudio --python examples/render_multiplane.py -- debug --out output/render_multiplane.jpg --out_blend output/render_multiplane.blend
```

![](output/render_multiplane.jpg)

## render skel

```bash
${blender} --background -noaudio --python examples/render_skel.py -- assets/thuman2-keypoints3d-000000.json --out output/render_skel.jpg --out_blend output/render_skel.blend
```

## render skel gt and pred

```bash
${blender} -noaudio --python examples/render_skel_gt_pred.py -- assets/s04_Hug1_000085.jpg.json
${blender} -noaudio --background --python examples/render_skel_gt_pred.py -- assets/field/s04_Hug\ 1_000070.jpg.json --no_pred --ground --grid assets/field/s04_Hug\ 1_000070_root.txt --out output/render_field.png --format PNG
```


## render animation example

```bash
${blender} -noaudio --python examples/animation/animate_ball.py
```