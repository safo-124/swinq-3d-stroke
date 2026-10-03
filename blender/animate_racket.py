"""Blender: animate a tennis racket from out/orientation.csv (sensor-only orientation).

Run (Blender 3.x / 4.x):
    blender --python blender/animate_racket.py
    blender --background --python blender/animate_racket.py -- --render out/blender_stroke.mp4

Builds a simple racket (handle, oval head, string bed, sensor marker) in the
sensor frame (X toward the butt cap, Y across the face, Z face normal), with the
origin at the throat, and keyframes its rotation quaternion from the bl_q*
columns (z-up world). The 0.96 s swing is slowed down SLOWMO times so it is
watchable; a head-path trail and a face-normal arrow are added.
"""
import csv
import math
import sys
from pathlib import Path

import bpy
from mathutils import Quaternion, Vector

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "out" / "orientation.csv"
FPS = 30
SLOWMO = 8                    # playback slow-down factor
IMU_HZ = 416.0
IMPACT_SAMPLE = 200

# racket geometry, metres, sensor frame (stage-3 refined)
BUTT_X, TIP_X = 0.231, -0.454
HEAD_CENTRE_X, HEAD_HALF_W = -0.272, 0.149
HEAD_HALF_L = HEAD_CENTRE_X - TIP_X


def read_orientations(path):
    """List of (sample, Quaternion) from the Blender-frame columns."""
    with open(path, newline="") as f:
        return [(int(r["sample"]), Quaternion((float(r["bl_qw"]), float(r["bl_qx"]),
                                               float(r["bl_qy"]), float(r["bl_qz"]))))
                for r in csv.DictReader(f)]


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def material(name, rgba, emission=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = rgba
    if emission:
        key = "Emission Color" if "Emission Color" in bsdf.inputs else "Emission"
        bsdf.inputs[key].default_value = rgba
        bsdf.inputs["Emission Strength"].default_value = emission
    return m


def build_racket():
    """Racket parented to an empty at the throat; returns the empty."""
    root = bpy.data.objects.new("Racket", None)
    bpy.context.collection.objects.link(root)
    root.rotation_mode = "QUATERNION"

    def add(obj, mat):
        obj.data.materials.append(mat)
        obj.parent = root
        return obj

    grip = material("Grip", (0.15, 0.08, 0.04, 1))
    frame = material("Frame", (0.05, 0.25, 0.8, 1))
    strings = material("Strings", (0.9, 0.9, 0.85, 0.6))
    sensor = material("Sensor", (1.0, 0.2, 0.1, 1), emission=3.0)

    # handle: cylinder along X from the butt to the start of the head
    handle_len = BUTT_X - (HEAD_CENTRE_X + HEAD_HALF_L)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.016, depth=handle_len,
                                        location=((BUTT_X + HEAD_CENTRE_X + HEAD_HALF_L) / 2, 0, 0),
                                        rotation=(0, math.pi / 2, 0))
    add(bpy.context.object, grip).name = "Handle"

    # head: torus squashed into an oval in the X-Y plane
    bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.012 / HEAD_HALF_W,
                                     location=(HEAD_CENTRE_X, 0, 0))
    head = add(bpy.context.object, frame)
    head.name = "Head"
    head.scale = (HEAD_HALF_L, HEAD_HALF_W, HEAD_HALF_W)

    # string bed: flat disc inside the head
    bpy.ops.mesh.primitive_circle_add(vertices=48, radius=1.0, fill_type="NGON",
                                      location=(HEAD_CENTRE_X, 0, 0))
    bed = add(bpy.context.object, strings)
    bed.name = "Strings"
    bed.scale = (HEAD_HALF_L * 0.95, HEAD_HALF_W * 0.95, 1)

    # sensor (SWINQ dampener) at the throat, on the string bed
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.015, location=(0, 0, 0.005))
    add(bpy.context.object, sensor).name = "Sensor"

    # face-normal arrow (+Z)
    bpy.ops.mesh.primitive_cone_add(radius1=0.012, depth=0.08, location=(HEAD_CENTRE_X, 0, 0.06))
    add(bpy.context.object, sensor).name = "FaceNormal"
    return root


def keyframe(root, data):
    """One keyframe per IMU sample on the slowed-down timeline."""
    scene = bpy.context.scene
    scene.render.fps = FPS
    frames_per_sample = FPS * SLOWMO / IMU_HZ
    for sample, q in data:
        root.rotation_quaternion = q
        root.keyframe_insert("rotation_quaternion", frame=1 + sample * frames_per_sample)
    scene.frame_start = 1
    scene.frame_end = int(1 + data[-1][0] * frames_per_sample) + 1
    impact = int(1 + IMPACT_SAMPLE * frames_per_sample)
    scene.timeline_markers.new("impact", frame=impact)
    try:                      # linear between samples (API differs across versions)
        for fc in root.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
    except AttributeError:
        bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"


def head_path(data):
    """Curve through the head tip over the whole swing."""
    tip = Vector((TIP_X, 0, 0))
    curve = bpy.data.curves.new("HeadPath", type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.003
    spline = curve.splines.new("POLY")
    spline.points.add(len(data) - 1)
    for p, (_, q) in zip(spline.points, data):
        v = q @ tip
        p.co = (v.x, v.y, v.z, 1)
    obj = bpy.data.objects.new("HeadPath", curve)
    obj.data.materials.append(material("Path", (0.6, 0.6, 0.6, 1)))
    bpy.context.collection.objects.link(obj)


def camera_and_light():
    bpy.ops.object.camera_add(location=(1.6, -1.6, 0.9))
    cam = bpy.context.object
    direction = Vector((0, 0, 0)) - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    bpy.ops.object.light_add(type="SUN", location=(2, -2, 4))
    bpy.context.object.data.energy = 3.0


def render(path):
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    scene.render.filepath = str(path)
    bpy.ops.render.render(animation=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    data = read_orientations(CSV_PATH)
    clear_scene()
    root = build_racket()
    keyframe(root, data)
    head_path(data)
    camera_and_light()
    if "--render" in argv:
        render(Path(argv[argv.index("--render") + 1]).resolve())


if __name__ == "__main__":
    main()
