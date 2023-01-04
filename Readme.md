# Blender

## install

```bash
<path_to_blender>/2.93/python/bin/python3.9 -m ensurepip --upgrade
# For example, in MacOS:
/Applications/Blender.app/Contents/Resources/2.82/python/bin/python3.7m -m ensurepip --upgrade
/Applications/Blender.app/Contents/Resources/2.82/python/bin/python3.7m setup.py develop
```

## render_example

```bash
${blender} -noaudio --python apps/blender/render_example.py
```

## render grid

```bash
${blender} -noaudio --python apps/blender/render_grid.py
```