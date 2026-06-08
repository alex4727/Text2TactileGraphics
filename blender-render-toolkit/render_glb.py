"""
Render a GLB mesh into a prebuilt Blender scene.

Loads ``config/render_template.blend`` (camera, lighting, and the ``MeshMaterial``
clay shader), imports a GLB, normalizes its size/orientation, applies the template
material, and renders a single PNG.

Usage:
    blender -b config/render_template.blend --python render_glb.py -- \\
        --glb_file /path/to/mesh.glb --output output/render.png
"""

import argparse
import math
import os
import sys

import bpy
from mathutils import Matrix, Vector


def get_args():
    """Parse command line arguments."""
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Render GLB files like the test workflow"
    )
    parser.add_argument(
        "--glb_file", type=str, required=True, help="Path to GLB file to render"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/glb_render.png",
        help="Output path for rendered image",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="config/render_template.blend",
        help="Blender template file",
    )
    parser.add_argument(
        "--start_rot_x",
        type=float,
        default=-90,
        help="X rotation in degrees (default: -90 for GLB files)",
    )
    parser.add_argument(
        "--start_rot_y",
        type=float,
        default=0,
        help="Y rotation in degrees (default: 0)",
    )
    parser.add_argument(
        "--start_rot_z",
        type=float,
        default=0,
        help="Z rotation in degrees (default: 0)",
    )
    parser.add_argument(
        "--target_size",
        type=float,
        default=2.2,
        help="Target size for mesh normalization (largest XY "
        "dimension after scaling; 2.2 frames a relief plate "
        "to fill the template camera)",
    )
    parser.add_argument(
        "--render_samples", type=int, default=None, help="Override render samples"
    )
    parser.add_argument(
        "--keep_materials",
        action="store_true",
        help="Keep the GLB's embedded materials (skip applying "
        "the template's MeshMaterial). Used for baselines "
        "that ship their own albedo / normal / roughness / "
        "metallic textures inside the GLB.",
    )
    parser.add_argument(
        "--center_mode",
        type=str,
        default="xyz_center",
        choices=("z_floor", "xyz_center"),
        help='How to position the normalized mesh. "z_floor" '
        "(default, original toolkit behaviour for upright "
        "lamp meshes) puts Z_min on the world floor. "
        '"xyz_center" centres the mesh\'s bbox at the world '
        "origin — better for flat-relief plates whose centre "
        "should land on the camera target.",
    )

    return parser.parse_args(argv)


def import_glb(filepath):
    """Import a GLB file and return the imported mesh object."""
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.import_scene.gltf(filepath=filepath)

    imported_objects = [
        obj for obj in bpy.context.selected_objects if obj.type == "MESH"
    ]

    if not imported_objects:
        print(f"Warning: No mesh objects found in {filepath}")
        return None

    # Join multiple objects if needed
    if len(imported_objects) > 1:
        bpy.context.view_layer.objects.active = imported_objects[0]
        for obj in imported_objects:
            obj.select_set(True)
        bpy.ops.object.join()
        obj = bpy.context.active_object
    else:
        obj = imported_objects[0]

    return obj


def normalize_mesh(
    obj, target_size=2.0, rotation_offset=(0, 0, 0), center_mode="z_floor"
):
    """
    Normalize mesh: apply rotation, scale uniformly, position in world.

    center_mode:
      'z_floor'    — centre XY at origin, Z_min at floor (original behaviour,
                     designed for the upright lamp test mesh).
      'xyz_center' — centre the bounding box at world origin (better for flat
                     relief plates whose centre should sit at the camera target).
    """
    # Apply the rotation directly to the mesh data. The glTF importer parents
    # the imported mesh under a container empty, so setting obj.rotation_euler
    # and calling transform_apply does NOT rotate the geometry in world space —
    # the rotation silently has no effect. Rotating the mesh data is
    # parent-independent and always takes effect.
    rot_mat = (
        Matrix.Rotation(math.radians(rotation_offset[2]), 4, "Z")
        @ Matrix.Rotation(math.radians(rotation_offset[1]), 4, "Y")
        @ Matrix.Rotation(math.radians(rotation_offset[0]), 4, "X")
    )
    obj.data.transform(rot_mat)
    obj.data.update()

    # Activate and select the object for the scale/location applies below.
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # Get bounding box after rotation
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()

    world_verts = [obj.matrix_world @ v.co for v in mesh.vertices]

    min_coords = Vector(
        (
            min(v.x for v in world_verts),
            min(v.y for v in world_verts),
            min(v.z for v in world_verts),
        )
    )
    max_coords = Vector(
        (
            max(v.x for v in world_verts),
            max(v.y for v in world_verts),
            max(v.z for v in world_verts),
        )
    )

    obj_eval.to_mesh_clear()

    dimensions = max_coords - min_coords
    center = (min_coords + max_coords) / 2

    print(f"  After rotation dimensions: {dimensions}")
    print(f"  After rotation center: {center}")

    # Scale uniformly based on max X/Y dimension
    max_xy = max(dimensions.x, dimensions.y)
    scale_factor = target_size / max_xy if max_xy > 0 else 1.0

    print(f"  Scale factor: {scale_factor}")

    obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Recalculate bounds after scaling
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()

    world_verts = [obj.matrix_world @ v.co for v in mesh.vertices]

    min_coords = Vector(
        (
            min(v.x for v in world_verts),
            min(v.y for v in world_verts),
            min(v.z for v in world_verts),
        )
    )
    max_coords = Vector(
        (
            max(v.x for v in world_verts),
            max(v.y for v in world_verts),
            max(v.z for v in world_verts),
        )
    )

    obj_eval.to_mesh_clear()

    center_new = (min_coords + max_coords) / 2

    if center_mode == "xyz_center":
        # Centre the bounding box at world origin — flat reliefs render with
        # their centre on the camera target.
        obj.location = Vector((-center_new.x, -center_new.y, -center_new.z))
    else:
        # Default 'z_floor': centre XY, drop Z_min onto the floor (lamp-style).
        obj.location = Vector((-center_new.x, -center_new.y, -min_coords.z))
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    final_dims = max_coords - min_coords
    print(f"  Final dimensions: {final_dims}")

    return obj


def apply_template_material(obj, material_name="MeshMaterial"):
    """Apply the template material to the mesh."""
    mat = bpy.data.materials.get(material_name)

    if mat is None:
        print(f"Warning: Material '{material_name}' not found")
        print(f"Available materials: {[m.name for m in bpy.data.materials]}")
        return False

    obj.data.materials.clear()
    obj.data.materials.append(mat)
    print(f"  Applied material: {material_name}")
    return True


def main():
    args = get_args()

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    template_path = args.template
    if not os.path.isabs(template_path):
        template_path = os.path.join(script_dir, template_path)

    glb_path = args.glb_file
    if not os.path.isabs(glb_path):
        glb_path = os.path.join(script_dir, glb_path)

    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(script_dir, output_path)

    print(f"Template: {template_path}")
    print(f"GLB file: {glb_path}")
    print(f"Output: {output_path}")
    print(f"Rotation X: {args.start_rot_x}°")

    # Open template
    bpy.ops.wm.open_mainfile(filepath=template_path)

    # Remove existing meshes (except background)
    background_meshes = ["Plane", "Sphere", "Backdrop"]
    meshes_to_remove = []

    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name not in background_meshes:
            meshes_to_remove.append(obj.name)

    for name in meshes_to_remove:
        obj = bpy.data.objects.get(name)
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)
            print(f"  Removed: {name}")

    # Import GLB
    obj = import_glb(glb_path)
    if obj is None:
        print("ERROR: Failed to import GLB")
        return

    print(f"\nImported: {obj.name}")

    # Normalize (rotate, scale, position) - this is the key step
    normalize_mesh(
        obj,
        target_size=args.target_size,
        rotation_offset=(args.start_rot_x, args.start_rot_y, args.start_rot_z),
        center_mode=args.center_mode,
    )

    # Apply material — replace the imported GLB's materials with the template's
    # MeshMaterial (pink-purple clay) for "ours". For baselines that ship their
    # own albedo/normal/roughness/metallic, pass --keep_materials to leave the
    # imported GLB's material slots untouched so their textures show through.
    if args.keep_materials:
        slot_names = [
            s.material.name if s.material else "<empty>" for s in obj.material_slots
        ]
        print(
            f"  Keeping GLB-embedded materials ({len(slot_names)} slot(s)): {slot_names}"
        )
    else:
        apply_template_material(obj, "MeshMaterial")

    # Override samples if specified
    if args.render_samples is not None:
        bpy.context.scene.cycles.samples = args.render_samples

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Render
    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)

    print(f"\nRendered to: {output_path}")


if __name__ == "__main__":
    main()
