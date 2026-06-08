# GLB Rendering Guide

This toolkit renders a GLB mesh into a prebuilt Blender scene so that every mesh
is framed, lit, and shaded consistently. It contains a single entry point,
`render_glb.py`, and a single scene template, `config/render_template.blend`.

> `render_glb.py` is modified from
> https://github.com/dunbar12138/blender-render-toolkit/tree/main

## Requirements

- **Blender 4.2+** (the template ships in Blender 4.5 format; tested with 4.5).
  Run Blender in background mode (`-b`) — no GUI needed.
- The template ships its own camera, lighting, a light backdrop `Plane`, and a
  `MeshMaterial` clay shader.

## What `render_glb.py` does

1. Opens the scene template (`--template`, default `config/render_template.blend`).
2. Removes any placeholder meshes from the template (keeps `Plane`/`Sphere`/`Backdrop`).
3. Imports the GLB and joins multiple objects into one if needed.
4. Normalizes the mesh: applies a rotation offset, scales so its largest XY
   dimension equals `--target_size`, and positions it (`--center_mode`).
5. Applies the template's `MeshMaterial` (unless `--keep_materials` is passed).
6. Renders a single PNG to `--output`.

## Quick start

```bash
blender -b config/render_template.blend --python render_glb.py -- \
    --glb_file /path/to/mesh.glb \
    --output output/render.png
```

The `--` separates Blender's own arguments from the script's arguments; everything
after it is parsed by `render_glb.py`.

## Example

A sample mesh and its render ship in `examples/`:

- **Input mesh:** `examples/full_mesh.glb`
- **Render output:** `examples/example_render.png`

Reproduce the example render with:

```bash
blender -b config/render_template.blend --python render_glb.py -- \
    --glb_file examples/full_mesh.glb \
    --output examples/example_render.png \
    --render_samples 128
```

## Arguments

| flag | default | meaning |
|---|---|---|
| `--glb_file` | *(required)* | Path to the GLB mesh to render. |
| `--output` | `output/glb_render.png` | Output PNG path. |
| `--template` | `config/render_template.blend` | Blender scene to load. |
| `--start_rot_x` | `-90` | X rotation (degrees) applied before normalization. The default turns a Y-up relief plate to face the camera. |
| `--start_rot_y` | `0` | Y rotation (degrees). |
| `--start_rot_z` | `0` | Z rotation (degrees). |
| `--target_size` | `2.2` | Largest XY dimension after uniform scaling. |
| `--render_samples` | *(template value)* | Override Cycles sample count. |
| `--keep_materials` | off | Keep the GLB's embedded materials instead of applying `MeshMaterial`. |
| `--center_mode` | `xyz_center` | `xyz_center`: center the bounding box at the origin (flat-relief plates, default). `z_floor`: drop the mesh's Z-min onto the floor (upright meshes). |

Relative paths for `--glb_file`, `--output`, and `--template` are resolved against
the directory containing `render_glb.py`.

## Common adjustments

**Orientation.** GLB files are usually Y-up. The default `--start_rot_x -90`
rotates a relief plate so its detailed face points at the template camera (which
looks straight down the world -Z axis). Adjust `--start_rot_x/y/z` if a mesh needs
a different orientation.

**Framing.** Increase `--target_size` to make the mesh larger in frame, decrease
it to make it smaller.

**Baseline meshes with their own textures.** Pass `--keep_materials` to leave the
GLB's albedo / normal / roughness / metallic slots untouched instead of replacing
them with the template's clay shader.

**Render quality.** Pass `--render_samples` to override the template's Cycles
sample count (lower = faster preview, higher = cleaner final render).
