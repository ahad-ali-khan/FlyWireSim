from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coevolution import AdaptivePolicy, SpikeSensors
from rl_tabula_rasa.environment_config import (
    ATMOSPHERE_ENABLED, FOG_ANISOTROPY, FOG_DENSITY, FOG_END_DISTANCE,
    FOG_START_DISTANCE, JAW_CONTACT_RADIUS, OBSTACLE_DENSITY,
    SLOWMO_FOG_DENSITY_MULTIPLIER, SLOW_MOTION_DURATION,
    SLOW_MOTION_NEAR_MISS_DISTANCE, SLOW_MOTION_SPEED,
    TERRAIN_DISPLACEMENT_SCALE, TERRAIN_NOISE_FREQUENCY, TREE_BRANCH_COUNT,
    TREE_BRANCH_LENGTH, TREE_BRANCH_RADIUS, TREE_TRUNK_HEIGHT,
    TREE_TRUNK_RADIUS, sample_obstacle_sites,
)


ROOT = Path(__file__).resolve().parents[1]
BAT_BLEND = ROOT / "assets/models/creatures/bat/vampire/bat_packed.blend"
MOTH_BLEND = ROOT / "assets/models/creatures/moth/moth.blend"
SIGNAL_JSON = ROOT / "data/brain_signal.json"
LEARNING_JSON = Path(os.environ.get("FLYWIRE_LEARNING_STATE", ROOT / "data/learning_state_coevolution.json"))
RESULTS_JSON = os.environ.get("FLYWIRE_RESULTS_FILE")
REPLAY_FILE = os.environ.get("FLYWIRE_REPLAY_FILE")
LIVE_STATE_FILE = os.environ.get("FLYWIRE_LIVE_STATE_FILE")
LIVE_STREAM_FILE = os.environ.get("FLYWIRE_LIVE_STREAM_FILE")
VIEWER_MODE = os.environ.get("FLYWIRE_VIEWER_MODE", "live" if LIVE_STATE_FILE else "replay")
REPLAY_SPEED = max(0.05, float(os.environ.get("FLYWIRE_REPLAY_SPEED", "1.0")))
CAMERA_PRESET = os.environ.get("FLYWIRE_CAMERA_PRESET", "chase").lower()
HUD_MODE = os.environ.get("FLYWIRE_HUD_MODE", "full-data").lower()
SEED = int(os.environ["FLYWIRE_SEED"]) if "FLYWIRE_SEED" in os.environ else random.SystemRandom().randrange(2**32)
ARENA_FLOOR_Z = 0.65
ARENA_CEILING_Z = 2.8
BOUNDARY_FALLOFF = 0.55
BOUNDARY_K = 0.012
MAX_BOUNDARY_FORCE = 0.28
ARENA_BORDER_INSET = 0.315
SONAR_PULSE_DURATION = 12
SONAR_BASE_RADIUS = 5.1
SONAR_BRAIN_GAIN = 0.5
SONAR_OMNIDIRECTIONAL = True
SONAR_CONE_HALF_ANGLE = math.radians(80)
SHAKE_SPEED_THRESHOLD = 0.24
RENDER_CONFIG = {"resolution_x": 1280, "resolution_y": 720, "fps": 30,
                 "view_transform": "AgX", "exposure": 0.25}


def material(name: str, color: tuple[float, float, float, float], metallic=0.0, roughness=0.6):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def point_camera(camera, target):
    camera.rotation_euler = (Vector(target) - camera.location).to_track_quat("-Z", "Y").to_euler()


def loop_action(armature, action):
    armature.animation_data_clear()
    tracks = armature.animation_data_create().nla_tracks
    strip = tracks.new().strips.new(action.name, 1, action)
    strip.action_frame_start = action.frame_range[0]
    strip.action_frame_end = action.frame_range[1]
    strip.frame_end = bpy.context.scene.frame_end + 1
    strip.repeat = max(
        1.0,
        (strip.frame_end - strip.frame_start)
        / max(1.0, action.frame_range[1] - action.frame_range[0]),
    )
    strip.extrapolation = "NOTHING"


def clamp(value, low, high):
    return max(low, min(high, value))


def direction(vector):
    return vector.normalized() if vector.length > 1e-6 else Vector((0.0, 0.0, 0.0))


def mouth_contact(mouth, moth):
    return (mouth - moth).length < JAW_CONTACT_RADIUS


def inside_arena(position):
    return (abs(position.x) <= 3.35001 and abs(position.y) <= 2.45001
            and ARENA_FLOOR_Z - 1e-5 <= position.z <= ARENA_CEILING_Z + 1e-5)


def boundary_repulsion(position, half_x, half_y):
    """One inverse-square box-boundary force for bat and moth alike."""
    force = Vector((0.0, 0.0, 0.0))
    for axis, low, high in ((0, -half_x, half_x), (1, -half_y, half_y),
                            (2, ARENA_FLOOR_Z, ARENA_CEILING_Z)):
        for distance, sign in ((position[axis] - low, 1), (high - position[axis], -1)):
            if distance < BOUNDARY_FALLOFF:
                force[axis] += sign * min(MAX_BOUNDARY_FORCE, BOUNDARY_K / max(0.05, distance) ** 2)
    return force


def sight_line(name, color):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.008
    curve.bevel_resolution = 1
    segment = curve.splines.new("POLY")
    segment.points.add(1)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material(name + "_Mat", color))
    return segment


def set_line(segment, start, end):
    segment.points[0].co = (*start, 1.0)
    segment.points[1].co = (*end, 1.0)


def flame_tongue(name, location, height, width, bend, color, strength):
    vertices = []
    radii = (0.10, 0.42, 0.78, 1.0, 0.96, 0.86, 0.72, 0.57, 0.42,
             0.29, 0.18, 0.09, 0.01)
    for level, radius in enumerate(radii):
        t = level / (len(radii) - 1)
        for side in range(18):
            angle = side * math.tau / 18
            uneven = 1.0 + 0.075 * math.sin(angle * 3 + t * 8) + 0.04 * math.cos(angle * 5 - t * 6)
            vertices.append((bend * t * t + math.cos(angle) * width * radius * uneven,
                             math.sin(angle) * width * radius * 0.72 * uneven,
                             height * t))
    faces = []
    for level in range(len(radii) - 1):
        for side in range(18):
            a = level * 18 + side
            b = level * 18 + (side + 1) % 18
            faces.append((a, b, b + 18, a + 18))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    mat = material(name + "_glow", (*color, 1), roughness=0.25)
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = strength
    texture = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    height_ramp = nodes.new("ShaderNodeValToRGB")
    height_ramp.color_ramp.elements[0].position = 0.0
    height_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1)
    height_ramp.color_ramp.elements[1].position = 0.38
    height_ramp.color_ramp.elements[1].color = (0.85, 0.85, 0.85, 1)
    tip = height_ramp.color_ramp.elements.new(1.0)
    tip.color = (0.0, 0.0, 0.0, 1)
    flame_color = nodes.new("ShaderNodeValToRGB")
    flame_color.color_ramp.elements[0].position = 0.0
    flame_color.color_ramp.elements[0].color = (1.0, 0.93, 0.42, 1)
    flame_color.color_ramp.elements[1].position = 1.0
    flame_color.color_ramp.elements[1].color = (0.62, 0.025, 0.001, 1)
    mid_color = flame_color.color_ramp.elements.new(0.40)
    mid_color.color = (*color, 1)
    fresnel = nodes.new("ShaderNodeFresnel")
    inverse_edge = nodes.new("ShaderNodeMath")
    inverse_edge.operation = "SUBTRACT"
    inverse_edge.inputs[0].default_value = 1.0
    alpha = nodes.new("ShaderNodeMath")
    alpha.operation = "MULTIPLY"
    mix = nodes.new("ShaderNodeMixShader")
    links = mat.node_tree.links
    links.new(texture.outputs["Generated"], separate.inputs[0])
    links.new(separate.outputs["Z"], height_ramp.inputs[0])
    links.new(separate.outputs["Z"], flame_color.inputs[0])
    links.new(flame_color.outputs["Color"], emission.inputs["Color"])
    links.new(fresnel.outputs[0], inverse_edge.inputs[1])
    links.new(height_ramp.outputs["Color"], alpha.inputs[0])
    links.new(inverse_edge.outputs[0], alpha.inputs[1])
    links.new(alpha.outputs[0], mix.inputs[0])
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(emission.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs["Surface"])
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    obj.data.materials.append(mat)
    for face in mesh.polygons:
        face.use_smooth = True
    return obj


def terrain_height(x, y):
    frequency = TERRAIN_NOISE_FREQUENCY
    return TERRAIN_DISPLACEMENT_SCALE * (
        0.5 * math.sin(x * frequency) * math.cos(y * frequency * 0.83)
        + 0.5 * math.sin((x + y) * frequency * 0.57))


def _tube_mesh(vertices, faces, start, end, radius, sides=8):
    axis = Vector(end) - Vector(start)
    direction = axis.normalized()
    basis = direction.cross(Vector((0, 1, 0)))
    if basis.length < 1e-5:
        basis = direction.cross(Vector((1, 0, 0)))
    basis.normalize()
    other = direction.cross(basis).normalized()
    first = len(vertices)
    for point in (Vector(start), Vector(end)):
        for index in range(sides):
            angle = math.tau * index / sides
            offset = radius * (math.cos(angle) * basis + math.sin(angle) * other)
            vertices.append(tuple(point + offset))
    for index in range(sides):
        nxt = (index + 1) % sides
        faces.append((first + index, first + nxt, first + sides + nxt, first + sides + index))


def build_forest():
    vertices, faces = [], []
    _tube_mesh(vertices, faces, (0, 0, 0), (0, 0, TREE_TRUNK_HEIGHT), TREE_TRUNK_RADIUS)
    for branch in range(TREE_BRANCH_COUNT):
        theta = math.pi * 0.55 * branch
        start = Vector((0, 0, TREE_TRUNK_HEIGHT * (0.60 + 0.13 * branch)))
        end = start + Vector((math.cos(theta), math.sin(theta), 0.55)).normalized() * TREE_BRANCH_LENGTH
        _tube_mesh(vertices, faces, start, end, TREE_BRANCH_RADIUS)
    mesh = bpy.data.meshes.new("Forest_Tree_Prototype_Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh


def instance_forest(points, scene):
    point_mesh = bpy.data.meshes.new("Forest_Instance_Points_Mesh")
    point_mesh.from_pydata([(x, y, terrain_height(x, y)) for x, y, _ in points], [], [])
    point_mesh.update()
    point_obj = bpy.data.objects.new("Forest_GeometryNodes_Instances", point_mesh)
    bpy.context.collection.objects.link(point_obj)

    bark = bpy.data.materials.new("Mossy_Bark")
    bark.diffuse_color = (0.12, 0.19, 0.13, 1)
    bark.use_nodes = True
    nodes, links = bark.node_tree.nodes, bark.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 5.0
    colors = nodes.new("ShaderNodeValToRGB")
    colors.color_ramp.elements[0].color = (0.035, 0.07, 0.045, 1)
    colors.color_ramp.elements[1].color = (0.24, 0.29, 0.17, 1)
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.22
    bump.inputs["Distance"].default_value = 0.08
    links.new(noise.outputs["Fac"], colors.inputs["Fac"])
    links.new(colors.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    tree_mesh = build_forest()
    tree_mesh.materials.append(bark)
    for polygon in tree_mesh.polygons:
        polygon.use_smooth = True
    prototype = bpy.data.objects.new("Forest_Tree_Prototype", tree_mesh)
    bpy.context.collection.objects.link(prototype)
    prototype.hide_set(True)

    group = bpy.data.node_groups.new("Forest_Tree_Instances", "GeometryNodeTree")
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    nodes, links = group.nodes, group.links
    input_node = nodes.new("NodeGroupInput")
    output_node = nodes.new("NodeGroupOutput")
    object_info = nodes.new("GeometryNodeObjectInfo")
    object_info.inputs["Object"].default_value = prototype
    object_info.transform_space = "ORIGINAL"
    object_info.inputs["As Instance"].default_value = True
    instances = nodes.new("GeometryNodeInstanceOnPoints")
    rotation = nodes.new("FunctionNodeRandomValue")
    rotation.data_type = "FLOAT_VECTOR"
    rotation.inputs["Min"].default_value = (0, 0, 0)
    rotation.inputs["Max"].default_value = (0, 0, math.tau)
    rotation.inputs["Seed"].default_value = SEED % (2**31)
    links.new(input_node.outputs["Geometry"], instances.inputs["Points"])
    links.new(object_info.outputs["Geometry"], instances.inputs["Instance"])
    links.new(rotation.outputs["Value"], instances.inputs["Rotation"])
    links.new(instances.outputs["Instances"], output_node.inputs["Geometry"])
    modifier = point_obj.modifiers.new("Geometry Nodes • seeded forest", "NODES")
    modifier.node_group = group
    point_obj["density_per_m2"] = OBSTACLE_DENSITY
    point_obj["seed"] = SEED % (2**31)
    return point_obj


def add_atmosphere(scene, camera):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 2.5))
    volume = bpy.context.view_layer.objects.active
    volume.name = "Moonlit_Atmosphere"
    volume.scale = (8, 6.4, 5.2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    fog = bpy.data.materials.new("Atmospheric_Volume")
    fog.use_nodes = True
    nodes = fog.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeVolumePrincipled")
    principled.inputs["Color"].default_value = (0.27, 0.36, 0.52, 1)
    principled.inputs["Anisotropy"].default_value = FOG_ANISOTROPY
    position = nodes.new("ShaderNodeNewGeometry")
    distance = nodes.new("ShaderNodeVectorMath")
    distance.operation = "DISTANCE"
    nodes.new("ShaderNodeMapRange")
    map_range = next(node for node in nodes if node.bl_idname == "ShaderNodeMapRange")
    map_range.clamp = True
    map_range.inputs["From Min"].default_value = FOG_START_DISTANCE
    map_range.inputs["From Max"].default_value = FOG_END_DISTANCE
    map_range.inputs["To Min"].default_value = 0.0
    map_range.inputs["To Max"].default_value = 1.0
    distance.inputs[1].default_value = camera.location
    density = nodes.new("ShaderNodeMath")
    density.operation = "MULTIPLY"
    density.inputs[1].default_value = FOG_DENSITY
    fog.node_tree.links.new(position.outputs["Position"], distance.inputs[0])
    fog.node_tree.links.new(distance.outputs["Value"], map_range.inputs["Value"])
    fog.node_tree.links.new(map_range.outputs["Result"], density.inputs[0])
    fog.node_tree.links.new(density.outputs[0], principled.inputs["Density"])
    fog.node_tree.links.new(principled.outputs["Volume"], output.inputs["Volume"])
    volume.data.materials.append(fog)
    volume.hide_render = not ATMOSPHERE_ENABLED
    volume.hide_set(not ATMOSPHERE_ENABLED)
    return volume, density.inputs[1], distance.inputs[1]


def make_spark(name, color, strength, radius=0.025):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=radius)
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    mat = bpy.data.materials.get(f"{name}_Material") or bpy.data.materials.new(f"{name}_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1)
    emission.inputs["Strength"].default_value = strength
    mat.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
    obj.data.materials.append(mat)
    return obj


def setup():
    random.seed(SEED)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with bpy.data.libraries.load(str(MOTH_BLEND), link=False) as (data_from, data_to):
        data_to.objects = list(data_from.objects)
        data_to.actions = list(data_from.actions)
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
    moth_root = bpy.data.objects["Sketchfab_model"]
    moth_arm = bpy.data.objects["Object_12"]
    helper = bpy.data.objects.get("Icosphere")
    if helper is not None:
        bpy.data.objects.remove(helper, do_unlink=True)

    with bpy.data.libraries.load(str(BAT_BLEND), link=False) as (data_from, data_to):
        data_to.objects = ["Armature_Bat", "Bat_LP_Anim"]
        data_to.actions = ["Bat_Flying"]
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
    bat_arm = bpy.data.objects["Armature_Bat"]
    bat_mesh = bpy.data.objects["Bat_LP_Anim"]
    scene = bpy.context.scene
    live_world_state = {}
    if LIVE_STATE_FILE and Path(LIVE_STATE_FILE).exists():
        try:
            live_world_state = json.loads(Path(LIVE_STATE_FILE).read_text())
        except (OSError, json.JSONDecodeError):
            live_world_state = {}
    scene.frame_start = 1
    scene.frame_end = 480
    bpy.context.view_layer.objects.active = bat_arm
    bat_arm.select_set(True)
    bat_arm.scale = (0.18, 0.18, 0.18)
    bat_arm.location = (-2.6, -0.8, 1.65)
    bat_flying_action = bpy.data.actions["Bat_Flying"]
    loop_action(bat_arm, bat_flying_action)

    moth_root.scale = (0.00010, 0.00010, 0.00010)
    moth_root.location = (2.0, 0.8, 1.65)
    loop_action(moth_arm, bpy.data.actions["Flying"])
    moth_body = bpy.data.objects["Object_19"]

    def place_moth(position, velocity):
        # The inspected rig has zero root rotation and its body is longest on
        # local Y; the existing heading mapping therefore points +Y forward.
        old_yaw = moth_root.rotation_euler[2]
        yaw = math.atan2(-velocity.x, velocity.y) if velocity.length > 1e-6 else old_yaw
        turn = math.atan2(math.sin(yaw - old_yaw), math.cos(yaw - old_yaw))
        horizontal = math.hypot(velocity.x, velocity.y)
        moth_root.rotation_euler = (math.atan2(velocity.z, max(horizontal, 1e-6)),
                                    clamp(-turn * 1.8, -0.38, 0.38), yaw)
        moth_root.location = position
        bpy.context.view_layer.update()
        body_center = sum((moth_body.matrix_world @ Vector(corner) for corner in moth_body.bound_box), Vector()) / 8
        moth_root.location += position - body_center

    scene.render.fps = RENDER_CONFIG["fps"]
    scene.render.resolution_x = RENDER_CONFIG["resolution_x"]
    scene.render.resolution_y = RENDER_CONFIG["resolution_y"]
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = RENDER_CONFIG["exposure"]
    if hasattr(scene.render, "use_motion_blur"):
        scene.render.use_motion_blur = True
    scene.render.engine = "BLENDER_EEVEE" if "BLENDER_EEVEE" in {
        item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items
    } else "BLENDER_EEVEE_NEXT"
    scene.world = bpy.data.worlds.new("FlyWire World")
    scene.world.color = (0.06, 0.09, 0.14)
    scene.world.use_nodes = True
    nodes = scene.world.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Color"].default_value = (0.035, 0.06, 0.10, 1)
    background.inputs["Strength"].default_value = 0.30
    scene.world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])

    ground_mat = material("Ground", (0.025, 0.07, 0.06, 1), roughness=0.85)
    vertices, faces = [], []
    x_count, y_count = 100, 76
    for j in range(y_count + 1):
        y = -3.6 + 7.2 * j / y_count
        for i in range(x_count + 1):
            x = -4.5 + 9.0 * i / x_count
            vertices.append((x, y, 0.0))
    for j in range(y_count):
        for i in range(x_count):
            a = j * (x_count + 1) + i
            faces.append((a, a + 1, a + x_count + 2, a + x_count + 1))
    ground_mesh = bpy.data.meshes.new("Forest_Floor_Grid")
    ground_mesh.from_pydata(vertices, [], faces)
    ground_mesh.update()
    ground = bpy.data.objects.new("Forest_Floor", ground_mesh)
    bpy.context.collection.objects.link(ground)
    terrain_texture = bpy.data.textures.new("Forest_Terrain_Noise", type="CLOUDS")
    terrain_texture.noise_scale = 1.0 / TERRAIN_NOISE_FREQUENCY
    displacement = ground.modifiers.new("Procedural_terrain_0.22m", "DISPLACE")
    displacement.texture = terrain_texture
    displacement.strength = TERRAIN_DISPLACEMENT_SCALE
    displacement.mid_level = 0.5
    displacement.direction = "Z"
    ground.modifiers.new("Terrain_smoothing", "SUBSURF").levels = 1
    ground.data.materials.append(ground_mat)
    ground_nodes = ground_mat.node_tree.nodes
    noise = ground_nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 9.0
    ramp = ground_nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.018, 0.055, 0.05, 1)
    ramp.color_ramp.elements[1].color = (0.07, 0.15, 0.10, 1)
    ground_mat.node_tree.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    ground_mat.node_tree.links.new(ramp.outputs["Color"], ground_nodes.get("Principled BSDF").inputs["Base Color"])
    saved_obstacles = live_world_state.get("environment", {}).get("obstacles", [])
    obstacle_points = ([(float(item["x"]), float(item["y"]), float(item.get("angle", 0.0)))
                        for item in saved_obstacles]
                       if saved_obstacles else sample_obstacle_sites(SEED))
    forest_instances = instance_forest(obstacle_points, scene)
    border_mat = material("Arena_Border", (0.12, 0.22, 0.19, 1), metallic=0.15, roughness=0.7)
    border_objects = []
    for x, y, sx, sy in [
        (0, -2.8, 3.7, 0.035), (0, 2.8, 3.7, 0.035),
        (-3.7, 0, 0.035, 2.8), (3.7, 0, 0.035, 2.8),
    ]:
        before = set(bpy.data.objects)
        bpy.ops.mesh.primitive_cube_add(location=(x, y, 0.04))
        border = next(obj for obj in bpy.data.objects if obj not in before)
        border.name = "Arena_Border"
        border.scale = (sx, sy, 0.035)
        border.data.materials.append(border_mat)
        border_objects.append(border)
    # Interior extents come from the visible box border rather than a
    # separately guessed radius. The inset keeps the meshes clear of the wall.
    arena_half_x = min(abs(b.location.x) - b.dimensions.x / 2
                       for b in border_objects if abs(b.location.x) > 1) - ARENA_BORDER_INSET
    arena_half_y = min(abs(b.location.y) - b.dimensions.y / 2
                       for b in border_objects if abs(b.location.y) > 1) - ARENA_BORDER_INSET

    lamp_target = Vector((0.9, 1.2, 1.65))
    flame_world = Vector((lamp_target.x, lamp_target.y, 1.44))
    lamp_mat = material("Lamp", (0.08, 0.07, 0.05, 1), metallic=0.25, roughness=0.4)
    before = set(bpy.data.objects)
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=0.07, depth=1.35, location=(lamp_target.x, lamp_target.y, 0.68))
    lamp_post = next(obj for obj in bpy.data.objects if obj not in before)
    lamp_post.name = "Flame_Lamp_Post"
    lamp_post.data.materials.append(lamp_mat)
    before = set(bpy.data.objects)
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.26, depth=0.08, location=(lamp_target.x, lamp_target.y, 0.06))
    lamp_base = next(obj for obj in bpy.data.objects if obj not in before)
    lamp_base.name = "Flame_Lamp_Base"
    lamp_base.data.materials.append(lamp_mat)
    flames = [
        flame_tongue("Flame_outer", flame_world, 0.58, 0.15, 0.10, (1.0, 0.11, 0.002), 1.2),
        flame_tongue("Flame_gold", flame_world + Vector((-0.035, 0.015, 0.01)),
                     0.47, 0.085, -0.065, (1.0, 0.48, 0.015), 1.8),
        flame_tongue("Flame_core", flame_world + Vector((0.015, -0.025, 0.015)),
                     0.33, 0.06, 0.025, (1.0, 0.89, 0.28), 2.2),
        flame_tongue("Flame_side", flame_world + Vector((0.06, 0.02, 0.01)),
                     0.37, 0.07, 0.09, (1.0, 0.25, 0.006), 1.4),
    ]
    flame_shape = [[vertex.co.copy() for vertex in tongue.data.vertices] for tongue in flames]
    before = set(bpy.data.objects)
    bpy.ops.object.light_add(type="POINT", location=flame_world)
    flame_light = next(obj for obj in bpy.data.objects if obj not in before)
    flame_light.name = "Moth_Flame_Light"
    flame_light.data.energy = 650
    flame_light.data.color = (1.0, 0.38, 0.07)
    flame_light.data.shadow_soft_size = 0.6
    echo_curve = bpy.data.curves.new("Bat_Sonar_Spherical_Shell", "CURVE")
    echo_curve.dimensions = "3D"
    echo_curve.resolution_u = 2
    echo_curve.bevel_depth = 0.001
    echo_curve.bevel_resolution = 2
    for plane in ("xy", "xz", "yz"):
        spline = echo_curve.splines.new("POLY")
        spline.points.add(63)
        for index, point in enumerate(spline.points):
            angle = math.tau * index / 64
            cosine, sine = math.cos(angle), math.sin(angle)
            point.co = ((cosine, sine, 0, 1) if plane == "xy" else
                        (cosine, 0, sine, 1) if plane == "xz" else
                        (0, cosine, sine, 1))
        spline.use_cyclic_u = True
    echo_pulse = bpy.data.objects.new("Bat_Echo_Pulse", echo_curve)
    bpy.context.collection.objects.link(echo_pulse)
    echo_pulse.name = "Bat_Echo_Pulse"
    echo_mat = material("Echo_Pulse", (0.01, 0.12, 0.20, 1), roughness=0.3)
    echo_nodes = echo_mat.node_tree.nodes
    echo_nodes.clear()
    echo_output = echo_nodes.new("ShaderNodeOutputMaterial")
    echo_emission = echo_nodes.new("ShaderNodeEmission")
    echo_emission.inputs["Color"].default_value = (0.015, 0.30, 0.48, 1)
    echo_emission.inputs["Strength"].default_value = 0.65
    echo_mat.node_tree.links.new(echo_emission.outputs[0], echo_output.inputs["Surface"])
    if hasattr(echo_mat, "surface_render_method"):
        echo_mat.surface_render_method = "DITHERED"
    echo_pulse.data.materials.append(echo_mat)
    moth_sight = sight_line("Moth_Visual_Threat", (0.3, 0.75, 1.0, 1))
    moth_light = sight_line("Moth_Light_Target", (1.0, 0.55, 0.10, 1))
    before = set(bpy.data.objects)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=1)
    capture_dot = next(obj for obj in bpy.data.objects if obj not in before)
    capture_dot.name = "Capture_Contact_Dot"
    capture_dot.scale = (0, 0, 0)
    dot_mat = material("Capture_Glow", (0.1, 1.0, 0.15, 1), roughness=0.2)
    dot_mat.node_tree.nodes.get("Principled BSDF").inputs["Emission Color"].default_value = (0.04, 1, 0.08, 1)
    dot_mat.node_tree.nodes.get("Principled BSDF").inputs["Emission Strength"].default_value = 8
    capture_dot.data.materials.append(dot_mat)
    before = set(bpy.data.objects)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=8, radius=1)
    echo_hit_dot = next(obj for obj in bpy.data.objects if obj not in before)
    echo_hit_dot.name = "Echo_Pulse_Hit_Dot"
    echo_hit_dot.scale = (0, 0, 0)
    hit_mat = material("Echo_Hit_Glow", (1.0, 0.04, 0.04, 1), roughness=0.2)
    hit_mat.node_tree.nodes.get("Principled BSDF").inputs["Emission Color"].default_value = (1.0, 0.01, 0.01, 1)
    hit_mat.node_tree.nodes.get("Principled BSDF").inputs["Emission Strength"].default_value = 10
    echo_hit_dot.data.materials.append(hit_mat)

    particle_rng = random.Random(SEED)
    dust_particles = []
    for particle_index in range(18):
        obj = make_spark(f"Moon_Dust_{particle_index:02d}", (0.45, 0.66, 0.85), 0.35, 0.018)
        base = Vector((particle_rng.uniform(-3.2, 3.2), particle_rng.uniform(-2.4, 2.4),
                       particle_rng.uniform(0.75, 2.7)))
        dust_particles.append((obj, base, particle_rng.random() * math.tau))
    ember_particles = [make_spark(f"Flame_Ember_{i:02d}", (1.0, 0.27, 0.025), 2.8, 0.022)
                       for i in range(8)]
    wing_particles = [make_spark(f"Moth_Scale_{i:02d}", (0.82, 0.68, 0.48), 0.7, 0.012)
                      for i in range(6)]
    impact_particles = [make_spark(f"Contact_Spark_{i:02d}", (1.0, 0.78, 0.38), 5.0, 0.035)
                        for i in range(12)]
    impact_state = {"ticks": 0, "center": Vector((0, 0, 0))}

    def animate_particles(frame, moth_position, flame_position, impact_position=None, hit=False):
        for obj, base, phase in dust_particles:
            obj.location = base + Vector((0.025 * math.sin(frame * 0.03 + phase),
                                          0.018 * math.cos(frame * 0.027 + phase),
                                          0.04 * math.sin(frame * 0.021 + phase)))
            size = 0.75 + 0.25 * math.sin(frame * 0.08 + phase)
            obj.scale = (size,) * 3
        for index, obj in enumerate(ember_particles):
            phase = index * math.tau / len(ember_particles) + frame * 0.11
            obj.location = flame_position + Vector((0.10 * math.cos(phase),
                0.07 * math.sin(phase), -0.32 + 0.45 * ((frame * 0.025 + index * 0.13) % 1)))
            obj.scale = (0.6 + 0.4 * math.sin(frame * 0.31 + index),) * 3
        for index, obj in enumerate(wing_particles):
            phase = index * math.tau / len(wing_particles) + frame * 0.19
            obj.location = moth_position + Vector((0.10 * math.cos(phase),
                0.10 * math.sin(phase), 0.045 * math.sin(phase * 2)))
            obj.scale = (0.7 + 0.3 * math.sin(frame * 0.3 + index),) * 3
        if hit and impact_position is not None:
            impact_state["ticks"] = 10
            impact_state["center"] = Vector(impact_position)
        ticks = impact_state["ticks"]
        for index, obj in enumerate(impact_particles):
            if ticks:
                phase = index * math.tau / len(impact_particles)
                direction = Vector((math.cos(phase), math.sin(phase), 0.35 * math.sin(phase * 2))).normalized()
                obj.location = impact_state["center"] + direction * (0.035 * (11 - ticks))
                obj.scale = (ticks / 10,) * 3
            else:
                obj.scale = (0.0,) * 3
        if ticks:
            impact_state["ticks"] -= 1

    for location, energy, size, color in [
        ((0, -4, 10), 550, 6, (1.0, 0.82, 0.66)),
        ((7, 4, 6), 620, 5, (0.55, 0.72, 1.0)),
        ((-7, -2, 5), 480, 4, (0.65, 0.85, 1.0)),
    ]:
        before = set(bpy.data.objects)
        bpy.ops.object.light_add(type="AREA", location=location)
        light = next(obj for obj in bpy.data.objects if obj not in before)
        light.data.energy = energy
        light.data.shape = "DISK"
        light.data.size = size
        light.data.color = color
        point_camera(light, (0, 0, 1.2))

    before = set(bpy.data.objects)
    bpy.ops.object.camera_add(location=(0.0, -6.5, 7.2))
    camera = next(obj for obj in bpy.data.objects if obj not in before)
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 7.5
    point_camera(camera, (0, 0, 1.55))
    scene.camera = camera
    atmosphere, fog_density_socket, fog_camera_socket = add_atmosphere(scene, camera)

    if REPLAY_FILE or LIVE_STATE_FILE:
        live_mode = bool(LIVE_STATE_FILE and not REPLAY_FILE)
        replay = {} if live_mode else json.loads(Path(REPLAY_FILE).read_text())
        frames = replay.get("frames", [])
        if not live_mode and not frames:
            raise ValueError("replay has no frames")
        live_cache = {"record": None, "last_warning": 0.0, "mtime": None,
                      "stream": None, "stream_path": None, "frame_count": 0}
        curve_file = Path(os.environ.get("FLYWIRE_LEARNING_CURVE_FILE", str(ROOT / "results/phase0_learning_curve.json")))
        curve_data = json.loads(curve_file.read_text()).get("curve", []) if curve_file.exists() else []
        curve_data = curve_data[::max(1, len(curve_data) // 90)]
        echo_pulse.hide_render = not live_mode
        echo_pulse.hide_set(not live_mode)
        echo_hit_dot.hide_render = True
        echo_hit_dot.hide_set(True)
        before = set(bpy.data.objects)
        bpy.ops.object.light_add(type="POINT", location=(0, 0, 3))
        bat_follow_light = next(obj for obj in bpy.data.objects if obj not in before)
        bat_follow_light.name = "Cinematic_Bat_Fill"
        bat_follow_light.data.energy = 280
        bat_follow_light.data.color = (0.62, 0.79, 1.0)
        bat_follow_light.data.shadow_soft_size = 1.2
        before = set(bpy.data.objects)
        bpy.ops.object.light_add(type="POINT", location=(0, 0, 3))
        moth_follow_light = next(obj for obj in bpy.data.objects if obj not in before)
        moth_follow_light.name = "Cinematic_Moth_Fill"
        moth_follow_light.data.energy = 90
        moth_follow_light.data.color = (1.0, 0.78, 0.54)
        moth_follow_light.data.shadow_soft_size = 0.8
        flame_offsets = [tongue.location.copy() - flame_world for tongue in flames]
        replay_status = {"episode": 1, "tick": 0, "outcome": "hunting", "contact": False,
                         "distance": 0.0, "index": 0, "role": "bat", "stale": False,
                         "near_miss_active": False, "slowmo_remaining": 0,
                         "sonar_occluded": False, "vision_occluded": False,
                         "bat_sonar_hit": False, "bat_alignment": None,
                         "bat_alignment_reward": 0.0, "bat_aligned_thrust_reward": 0.0}

        def read_live_record():
            path = Path(LIVE_STATE_FILE)
            now = time.monotonic()
            stream_path = Path(LIVE_STREAM_FILE) if LIVE_STREAM_FILE else None
            if stream_path and stream_path.exists():
                try:
                    if (live_cache["stream"] is None
                            or live_cache["stream_path"] != stream_path
                            or stream_path.stat().st_size < live_cache["stream"].tell()):
                        if live_cache["stream"] is not None:
                            live_cache["stream"].close()
                        live_cache["stream"] = stream_path.open("r", encoding="utf-8")
                        live_cache["stream_path"] = stream_path
                        live_cache["frame_count"] = 0
                    handle = live_cache["stream"]
                    offset = handle.tell()
                    line = handle.readline()
                    if line and line.endswith("\n"):
                        record = json.loads(line)
                        live_cache["record"] = record
                        live_cache["frame_count"] += 1
                        replay_status["buffered_frame"] = live_cache["frame_count"]
                        replay_status["stale"] = False
                        replay_status["buffered_playback"] = True
                        return record
                    if not line and live_cache["record"] and live_cache["record"].get("playback_kind") == "evaluation":
                        handle.seek(0)
                        record = json.loads(handle.readline())
                        live_cache["record"] = record
                        live_cache["frame_count"] = 1
                        return record
                    if line:
                        handle.seek(offset)
                    modified = path.stat().st_mtime
                    replay_status["stale"] = time.time() - modified > 5.0
                    replay_status["buffered_playback"] = True
                    return live_cache["record"]
                except (OSError, json.JSONDecodeError):
                    replay_status["stale"] = True
                    replay_status["buffered_playback"] = True
                    return live_cache["record"]
            try:
                modified = path.stat().st_mtime
                record = json.loads(path.read_text())
                live_cache["record"] = record
                live_cache["mtime"] = modified
                replay_status["stale"] = time.time() - modified > 5.0
                replay_status["buffered_playback"] = False
            except (OSError, json.JSONDecodeError):
                replay_status["stale"] = True
            if replay_status["stale"] and now - live_cache["last_warning"] > 10.0:
                print(f"Live state missing or stale; holding last available viewer state: {path}", flush=True)
                live_cache["last_warning"] = now
            return live_cache["record"]

        def draw_replay_hud():
            region = bpy.context.region
            if region is None:
                return
            shader = gpu.shader.from_builtin("UNIFORM_COLOR")
            top = region.height - 85
            blf.size(0, 16)
            mode_label = (("SAC EVALUATION REPLAY  /  frozen policies" if replay_status.get("evaluation") else
                           "BUFFERED TRAINING PLAYBACK  /  recorded frames") if live_mode else
                          f"PHASE {replay.get('policy_phase', '?')} PPO REPLAY  /  continuous yaw • pitch • thrust")
            lines = ([
                f"FLYWIRE SIM  /  {replay_status['outcome'].upper()}",
                f"Episode {replay_status['episode']}  •  Tick {replay_status['tick']}  •  "
                f"Range {replay_status['distance']:.2f}m",
            ] if HUD_MODE == "cinematic" else [
                mode_label,
                f"Episode {replay_status['episode']}  Tick {replay_status['tick']}  "
                  f"Range {replay_status['distance']:.2f}m",
                f"Result: {replay_status['outcome']}  |  "
                  f"Swept jaw contact: {'YES' if replay_status['contact'] else 'NO'}",
                f"Slow motion: {replay_status['slowmo_remaining']} ticks  |  "
                f"Occlusion: sonar {'blocked' if replay_status['sonar_occluded'] else 'clear'}, "
                f"vision {'blocked' if replay_status['vision_occluded'] else 'clear'}",
                (f"Bat aim: {replay_status['bat_alignment']:+.2f}  |  "
                 f"alignment shaping: {replay_status['bat_alignment_reward']:+.3f}"
                 if replay_status["bat_sonar_hit"] else
                 f"Bat aim: no sonar hit | shaping: {replay_status['bat_alignment_reward']:+.3f}"),
                (f"Recorded evaluation / moth: {replay_status.get('opponent_mode', 'unknown')} / no updates" if replay_status.get("evaluation") else
                 f"Currently training: {replay_status['role']}  |  Live state: "
                 f"{'STALE — holding last state' if replay_status['stale'] else ('buffered frame ' + str(replay_status.get('buffered_frame', 0)) if replay_status.get('buffered_playback') else 'updating')}" if live_mode else
                 ("Moth: FlyWire-seeded network (trained)" if replay.get("connectome_seeded_moth")
                  else "Moth: random-init learned policy")),
                "FlyWire/Brian2 recorded output: separate prerecorded SEZ; not policy weights.",
            ])
            if (HUD_MODE != "cinematic" and live_mode and replay_status.get("phase") == 9
                    and replay_status.get("workspace_weights")):
                lines.extend((
                    "PHASE 9  /  learned 4-specialist workspace weights",
                    *(f"{role.title()}: " + "  ".join(
                        f"{name.replace('_', ' ')} {100.0 * float(weight):.0f}%"
                        for name, weight in replay_status["workspace_weights"].get(role, {}).items())
                      for role in ("bat", "moth")),
                    "Self-model: own next-position prediction  |  Belief: estimated opponent position",
                ))
                for role, label in (("bat", "Bat belief → moth"), ("moth", "Moth belief → bat")):
                    position = replay_status.get("belief_estimate", {}).get(role)
                    if position is not None:
                        lines.append(f"{label}: ({position[0]:+.2f}, {position[1]:+.2f}, {position[2]:+.2f}) m")
            panel_height = max(145, 34 + len(lines) * 23)
            batch = batch_for_shader(shader, "TRI_FAN", {"pos": [
                (20, top - panel_height), (590, top - panel_height), (590, top), (20, top)]})
            gpu.state.blend_set("ALPHA")
            shader.bind()
            shader.uniform_float("color", (0.025, 0.045, 0.065, 0.90))
            batch.draw(shader)
            gpu.state.blend_set("NONE")
            for i, line in enumerate(lines):
                blf.position(0, 35, top - 26 - i * 23, 0)
                blf.color(0, 1, 0.84 if i == 0 else 0.95, 0.55 if i == 0 else 1, 1)
                blf.draw(0, line)
            if curve_data:
                graph_panel = batch_for_shader(shader, "TRI_FAN", {"pos": [
                    (20, 25), (590, 25), (590, 137), (20, 137)]})
                gpu.state.blend_set("ALPHA")
                shader.bind()
                shader.uniform_float("color", (0.025, 0.045, 0.065, 0.90))
                graph_panel.draw(shader)
                gpu.state.blend_set("NONE")
                blf.size(0, 13)
                blf.position(0, 35, 112, 0)
                blf.color(0, 0.76, 0.89, 1.0, 1)
                blf.draw(0, "TRAINING CURVE  /  catch rate, last 100 hunts")
                total = max(1, curve_data[-1]["step"] - curve_data[0]["step"])
                coords = [(36 + 535 * (p["step"] - curve_data[0]["step"]) / total,
                           42 + 62 * p["catch_rate_last_100"]) for p in curve_data]
                line_batch = batch_for_shader(shader, "LINE_STRIP", {"pos": coords})
                shader.bind()
                shader.uniform_float("color", (0.16, 0.78, 1.0, 1.0))
                line_batch.draw(shader)

        def update_replay(index):
            if live_mode:
                record = read_live_record()
                if not record:
                    replay_status.update({"outcome": "waiting for trainer", "stale": True})
                    return
                bat_state, moth_state = record["bat"], record["moth"]
                bat_position, moth_position = Vector(bat_state["position"]), Vector(moth_state["position"])
                bat_speed, moth_speed = Vector(bat_state["velocity"]), Vector(moth_state["velocity"])
                target = Vector(moth_state["flame_position"])
                jaw_position = Vector(bat_state.get("jaw_position", bat_position))
                caught = bool(record.get("caught", False))
                contact_position = record.get("jaw_contact_position")
                distance = float(record.get("jaw_distance", (jaw_position - moth_position).length))
                sonar_radius = float(bat_state.get("sonar_radius", 0.0))
                episode = int(record.get("episode", 0))
                tick = int(record.get("tick", 0))
                phase_label = str(record.get("phase", "0"))
                phase_prefix = phase_label.split("-", 1)[0]
                phase = int(phase_prefix) if phase_prefix.isdigit() else 0
                outcome = "caught" if caught else "survived / timeout" if record.get("timeout") else "hunting"
                replay_status.update({"episode": episode, "tick": tick, "outcome": outcome,
                                      "evaluation": record.get("playback_kind") == "evaluation",
                                      "opponent_mode": record.get("moth_opponent_mode"),
                                      "distance": distance, "role": record.get("active_training_role", "bat"),
                                      "sonar_occluded": bool(record.get("sonar_occluded", False)),
                                      "vision_occluded": bool(record.get("vision_occluded", False)),
                                      "bat_sonar_hit": bool(record.get("bat_sonar_hit", False)),
                                      "bat_alignment": record.get("bat_alignment"),
                                      "bat_alignment_reward": float(record.get("bat_alignment_reward", 0.0)),
                                      "bat_aligned_thrust_reward": float(record.get(
                                          "bat_aligned_thrust_reward", 0.0)),
                                      "phase": phase,
                                      "workspace_weights": record.get("workspace_weights", {}),
                                      "self_model_prediction": record.get("self_model_predicted_position", {}),
                                      "belief_estimate": record.get("belief_state_estimated_position", {})})
            else:
                record = frames[index % len(frames)]
                bat_position = Vector(record["bat_position"])
                moth_position = Vector(record["moth_position"])
                bat_speed = Vector(record["bat_velocity"])
                moth_speed = Vector(record["moth_velocity"])
                target = Vector(record["flame_position"])
                sonar_radius = 0.0
                jaw_position = None
                caught = record["outcome"] == "caught_model_contact"
                contact_position = record.get("jaw_contact_position")
                distance = float(record["distance"])
                replay_status.update({"episode": record["episode"], "tick": record["tick"],
                                      "outcome": record["outcome"], "distance": distance,
                                      "index": index % len(frames),
                                      "sonar_occluded": bool(record.get("sonar_occluded", False)),
                                      "vision_occluded": bool(record.get("vision_occluded", False))})
            flame_z = max(0.52, target.z - 0.21)
            new_flame_world = Vector((target.x, target.y, flame_z))
            lamp_base.location = (target.x, target.y, 0.06)
            post_depth = max(0.38, flame_z - 0.08)
            lamp_post.location = (target.x, target.y, 0.08 + post_depth / 2)
            lamp_post.scale.z = post_depth / 1.35
            flame_light.location = new_flame_world
            flicker = 1.0 + 0.12 * math.sin(index * 0.37) + 0.05 * math.sin(index * 0.83)
            flame_light.data.energy = 650 * flicker
            for flame_index, (tongue, offset) in enumerate(zip(flames, flame_offsets)):
                tongue.location = new_flame_world + offset
                tongue.scale = (1.0 + 0.06 * math.sin(index * 0.29 + flame_index),
                                1.0 + 0.07 * math.sin(index * 0.37 + flame_index), flicker)
                tongue.rotation_euler[2] = 0.08 * math.sin(index * 0.21 + flame_index)
                for vertex, original in zip(tongue.data.vertices, flame_shape[flame_index]):
                    height_fraction = original.z / 0.58
                    vertex.co.x = original.x + 0.028 * height_fraction * math.sin(index * 0.32 + height_fraction * 8 + flame_index)
                    vertex.co.y = original.y + 0.017 * height_fraction * math.cos(index * 0.26 + height_fraction * 6 + flame_index)
            bat_arm.location = bat_position
            bat_follow_light.location = bat_position + Vector((-0.8, -0.7, 1.3))
            moth_follow_light.location = moth_position + Vector((0.35, -0.45, 0.65))
            if bat_speed.length > 1e-6:
                bat_arm.rotation_euler = (
                    math.atan2(bat_speed.z, max(math.hypot(bat_speed.x, bat_speed.y), 1e-6)),
                    0.0, math.atan2(-bat_speed.x, bat_speed.y),
                )
            place_moth(moth_position, moth_speed)
            if live_mode:
                echo_pulse.location = jaw_position
                echo_pulse.scale = (sonar_radius,) * 3
            # Cinematic replay omits the technical sight/attraction guide-lines.
            set_line(moth_sight, moth_position, moth_position)
            set_line(moth_light, moth_position, moth_position)
            scene.frame_set(1 + index % scene.frame_end)
            bpy.context.view_layer.update()
            jaw = bat_arm.pose.bones.get("Jaw_Lower")
            jaw_world = bat_arm.matrix_world @ jaw.tail if jaw else bat_position
            contact = caught and (contact_position is not None or
                                  mouth_contact(jaw_world, moth_position))
            replay_status["contact"] = contact
            animate_particles(index, moth_position, new_flame_world,
                              Vector(contact_position) if contact_position is not None else jaw_world, hit=caught)
            collision_distance = float(record.get("swept_jaw_distance", distance))
            collision_event = caught or collision_distance < SLOW_MOTION_NEAR_MISS_DISTANCE
            if collision_event and not replay_status["near_miss_active"]:
                replay_status["slowmo_remaining"] = SLOW_MOTION_DURATION
            replay_status["near_miss_active"] = collision_event
            if replay_status["slowmo_remaining"] > 0:
                fog_density_socket.default_value = FOG_DENSITY * SLOWMO_FOG_DENSITY_MULTIPLIER
            else:
                fog_density_socket.default_value = FOG_DENSITY
            capture_dot.location = (Vector(contact_position) if contact_position is not None
                                    else jaw_world)
            capture_dot.scale = (0.12,) * 3 if contact else (0.0,) * 3
            center = (bat_position + moth_position) / 2
            if CAMERA_PRESET == "wide":
                center = Vector((0, 0, 1.35))
                camera_offset, camera_scale = Vector((0.0, -9.0, 6.0)), 10.0
            elif CAMERA_PRESET == "hero":
                center = jaw_world
                camera_offset, camera_scale = Vector((0.0, -2.6, 1.0)), 3.2
            else:
                camera_offset, camera_scale = Vector((0.0, -4.8, 2.7)), None
            camera.location = center + camera_offset
            fog_camera_socket.default_value = camera.location
            point_camera(camera, center)
            camera.data.ortho_scale = (camera_scale if camera_scale is not None else
                clamp(2.4 + (bat_position - moth_position).length * 0.55, 3.5, 6.2))

        update_replay.save = lambda: None
        if not bpy.app.background:
            bpy.types.SpaceView3D.draw_handler_add(draw_replay_hud, (), "WINDOW", "POST_PIXEL")
            counter = {"index": 0}

            def show_replay_camera():
                for window in bpy.context.window_manager.windows:
                    screen = window.screen
                    for area in screen.areas:
                        if area.type != "VIEW_3D":
                            continue
                        area.spaces.active.shading.type = "MATERIAL"
                        area.spaces.active.overlay.show_overlays = False
                        area.spaces.active.region_3d.view_camera_zoom = 16
                        # Initialize the camera through Blender's operator, but
                        # do not toggle it off on the second timer invocation.
                        region = next((r for r in area.regions if r.type == "WINDOW"), None)
                        if region and area.spaces.active.region_3d.view_perspective != "CAMERA":
                            with bpy.context.temp_override(window=window, screen=screen,
                                                           area=area, region=region):
                                bpy.ops.view3d.view_camera()
                                area.spaces.active.show_region_toolbar = False
                return None

            def advance_replay():
                update_replay(counter["index"])
                counter["index"] += 1
                for window in bpy.context.window_manager.windows:
                    for area in window.screen.areas:
                        if area.type == "VIEW_3D":
                            area.tag_redraw()
                slowmo = SLOW_MOTION_SPEED if replay_status["slowmo_remaining"] > 0 else 1.0
                if replay_status["slowmo_remaining"] > 0:
                    replay_status["slowmo_remaining"] -= 1
                return 1 / (scene.render.fps * REPLAY_SPEED * slowmo)

            bpy.app.driver_namespace["flywiresim_live_update"] = update_replay
            bpy.app.driver_namespace["flywiresim_live_timer"] = advance_replay
            # Wait until Blender has initialized its UI areas. Changing region
            # visibility in the startup script can crash ED_area_init on macOS.
            bpy.app.timers.register(show_replay_camera, first_interval=1.0, persistent=True)
            bpy.app.timers.register(advance_replay, first_interval=0.1, persistent=True)
        else:
            if live_mode:
                update_replay(0)
                print(f"Live PPO viewer initialized; polling {LIVE_STATE_FILE}.", flush=True)
            else:
                verified = 0
                for frame_index in range(len(frames)):
                    update_replay(frame_index)
                    verified += int(replay_status["contact"])
                model_catches = sum(bool(item["caught"]) for item in replay.get("episodes", []))
                print(f"Animated jaw verification: {verified}/{model_catches} model catches")
        if not live_mode:
            print(f"PPO replay loaded: {len(frames)} frames from {REPLAY_FILE}")
        return update_replay

    signal = json.loads(SIGNAL_JSON.read_text()) if SIGNAL_JSON.exists() else [0.0] * 100
    saved = json.loads(LEARNING_JSON.read_text()) if LEARNING_JSON.exists() else {}
    rng = random.Random(SEED)

    def spawn_positions():
        for _ in range(100):
            bat = Vector((rng.uniform(-3.0, 3.0), rng.uniform(-2.0, 2.0), rng.uniform(1.0, 2.45)))
            moth = Vector((rng.uniform(-3.0, 3.0), rng.uniform(-2.0, 2.0), rng.uniform(1.0, 2.45)))
            if (bat - moth).length >= 2.5:
                return bat, moth
        return Vector((-2.6, -0.8, 1.65)), Vector((2.0, 0.8, 1.65))

    active = saved.get("active", {})
    def saved_vector(name, fallback):
        value = active.get(name)
        return Vector(value) if isinstance(value, list) and len(value) == 3 else fallback

    bat_pos, moth_pos = spawn_positions()
    bat_pos = saved_vector("bat_pos", bat_pos)
    moth_pos = saved_vector("moth_pos", moth_pos)
    spawn_bat, spawn_moth = bat_pos.copy(), moth_pos.copy()
    bat_learner = AdaptivePolicy(["DIRECT", "LEAD", "SWEEP"], saved.get("bat"), rng)
    moth_learner = AdaptivePolicy(["DODGE", "FLANK", "LATE", "FEINT"], saved.get("moth"), rng)
    bat_sensors = SpikeSensors(["echo"])
    moth_sensors = SpikeSensors(["vision", "flame"])
    flywire_history = deque([0.0] * 96, maxlen=96)
    catches = int(saved.get("catches", 0))
    escapes = int(saved.get("escapes", 0))
    episode = catches + escapes + 1
    recent = list(saved.get("recent", []))[-10:]
    episode_results = list(saved.get("episode_results", []))[-200:]
    tactic_switches = int(saved.get("tactic_switches", 0))
    total_ticks = int(saved.get("total_ticks", 0))
    attack_step = int(active.get("attack_step", 0))
    attack_limit = 230
    lamp_visit = bool(active.get("lamp_visit", False))
    last_brain = 0.0
    last_outcome = "HUNTING"
    bat_decision = "SEARCHING"
    moth_decision = "SEEKING LIGHT"
    echo_target = saved_vector("echo_target", None) if active.get("echo_target") else None
    moth_velocity = saved_vector("moth_velocity", Vector((0.0, 0.0, 0.0)))
    bat_velocity = saved_vector("bat_velocity", Vector((0.0, 0.0, 0.0)))
    feint_side = float(active.get("feint_side", 1.0))
    feint_triggered = bool(active.get("feint_triggered", False))
    event_timer = 0
    capture_timer = 0
    echo_hit_timer = 0
    last_distance = (moth_pos - bat_pos).length
    last_flame_distance = (moth_pos - lamp_target).length

    def features():
        distance = (moth_pos - bat_pos).length
        wall = min(3.35 - abs(moth_pos.x), 2.45 - abs(moth_pos.y),
                   moth_pos.z - ARENA_FLOOR_Z, ARENA_CEILING_Z - moth_pos.z)
        approach = clamp((last_distance - distance) * 8.0, -1.0, 1.0)
        bat_state = (1.0, clamp(1 - distance / 5.0, 0.0, 1.0),
                     clamp(moth_velocity.length * 8.0, 0.0, 1.0),
                     clamp(1 - wall / 1.4, 0.0, 1.0), clamp((last_brain + 1) / 2, 0.0, 1.0))
        moth_state = (1.0, clamp(1 - distance / 3.4, 0.0, 1.0),
                      clamp((approach + 1) / 2, 0.0, 1.0),
                      clamp(1 - wall / 1.4, 0.0, 1.0),
                      clamp(1 - (moth_pos - lamp_target).length / 3.0, 0.0, 1.0))
        return bat_state, moth_state

    bat_features, moth_features = features()
    for learner, feature, key in ((bat_learner, bat_features, "bat_current"),
                                  (moth_learner, moth_features, "moth_current")):
        current = active.get(key)
        if current in learner.actions:
            learner.current = current
            learner.previous_features = feature
        else:
            learner.choose(feature)

    def save_learning():
        snapshot = {
            "seed": SEED,
            "episodes": episode,
            "catches": catches,
            "escapes": escapes,
            "recent": recent[-10:],
            "tactic_switches": tactic_switches,
            "total_ticks": total_ticks,
            "episode_results": episode_results[-200:],
            "bat": bat_learner.state(),
            "moth": moth_learner.state(),
            "active": {
                "bat_pos": list(bat_pos), "moth_pos": list(moth_pos),
                "bat_velocity": list(bat_velocity), "moth_velocity": list(moth_velocity),
                "echo_target": list(echo_target) if echo_target else None,
                "attack_step": attack_step, "lamp_visit": lamp_visit,
                "feint_side": feint_side, "feint_triggered": feint_triggered,
                "bat_current": bat_learner.current, "moth_current": moth_learner.current,
            },
        }
        temporary = LEARNING_JSON.with_suffix(".tmp")
        temporary.write_text(json.dumps(snapshot, indent=2) + "\n")
        temporary.replace(LEARNING_JSON)
        if RESULTS_JSON:
            report = {
                "method": "online linear Q-learning; contextual tactics; hybrids unlocked from tested primitives",
                "seed": SEED, "ticks": total_ticks, "completed_episodes": catches + escapes,
                "catches": catches, "survivals": escapes, "tactic_switches": tactic_switches,
                "bat_policy": bat_learner.state(), "moth_policy": moth_learner.state(),
                "episodes": episode_results,
                "note": "FlyWire trace is fixed replay; spiking sensors and learned policies are simulated.",
            }
            Path(RESULTS_JSON).write_text(json.dumps(report, indent=2) + "\n")

    def reset_episode():
        nonlocal bat_pos, moth_pos, attack_step, lamp_visit
        nonlocal echo_target, moth_velocity, bat_velocity, feint_side, feint_triggered
        nonlocal last_distance, last_flame_distance
        nonlocal spawn_bat, spawn_moth
        bat_pos, moth_pos = spawn_positions()
        spawn_bat, spawn_moth = bat_pos.copy(), moth_pos.copy()
        attack_step = 0
        lamp_visit = False
        echo_target = None
        moth_velocity = Vector((0.0, 0.0, 0.0))
        bat_velocity = Vector((0.0, 0.0, 0.0))
        feint_side = random.choice((-1.0, 1.0))
        feint_triggered = False
        last_distance = (moth_pos - bat_pos).length
        last_flame_distance = (moth_pos - lamp_target).length
        bat_features, moth_features = features()
        bat_learner.choose(bat_features)
        moth_learner.choose(moth_features)
        bat_arm.location = bat_pos
        place_moth(moth_pos, moth_velocity)

    def draw_decision_feed():
        region = bpy.context.region
        if region is None:
            return
        panel_x, panel_top = 215, region.height - 72
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        corners = [
            (panel_x, panel_top - 660), (panel_x + 640, panel_top - 660),
            (panel_x + 640, panel_top), (panel_x, panel_top),
        ]
        batch = batch_for_shader(shader, "TRI_FAN", {"pos": corners})
        gpu.state.blend_set("ALPHA")
        shader.bind()
        shader.uniform_float("color", (0.025, 0.045, 0.065, 0.88))
        batch.draw(shader)
        gpu.state.blend_set("NONE")
        font_id = 0
        blf.size(font_id, 15)
        bat_state, moth_state = features()
        lines = [
            ("PREDATOR–PREY  /  LIVE LEARNING", (1.0, 0.82, 0.45, 1.0)),
            (f"EP {episode:03d}   HUNT CLOCK {(attack_limit - attack_step) / scene.render.fps:.1f}s   {last_outcome}", (1.0, 1.0, 1.0, 1.0)),
            (f"BAT   {bat_learner.current:<18} {bat_decision}", (1.0, 0.66, 0.54, 1.0)),
            (f"MOTH  {moth_learner.current:<18} {moth_decision}", (0.63, 0.85, 1.0, 1.0)),
            (f"Jaw range {(moth_pos - bat_pos).length:.2f}m   Moth→flame {(moth_pos - lamp_target).length:.2f}m", (0.85, 0.9, 0.93, 1.0)),
            (f"Catch {catches}   Survive {escapes}   Tactic switches {tactic_switches}   Recent {''.join(recent[-10:]) or '—'}", (0.85, 0.9, 0.93, 1.0)),
            ("TIMEOUT → BAT -1, MOTH +1   /   randomized starts", (0.86, 0.91, 0.67, 1.0)),
            ("Learned policy: contextual Q values  /  higher predicts better return", (1.0, 0.82, 0.45, 1.0)),
        ]
        lines += [
            (f"Bat   {name:<18} Q {bat_learner.value(name, bat_state):+0.2f}  used {bat_learner.counts[name]}",
             (1.0, 0.72, 0.65, 1.0)) for name in bat_learner.actions
        ]
        lines += [
            (f"Moth  {name:<18} Q {moth_learner.value(name, moth_state):+0.2f}  used {moth_learner.counts[name]}",
             (0.70, 0.86, 1.0, 1.0)) for name in moth_learner.actions
        ]
        lines.append((f"CYAN SPHERE bat sonar radius {SONAR_BASE_RADIUS + SONAR_BRAIN_GAIN * last_brain:.1f}m    BLUE moth vision → bat", (0.68, 0.86, 0.96, 1.0)))
        lines.append(("AMBER moth → flame    RED dot = sonar hit    GREEN dot = jaw contact", (1.0, 0.77, 0.48, 1.0)))
        lines.append(("FlyWire/Brian2 recorded output = fixed replay; sensors + policy = simulated", (0.70, 0.73, 0.75, 1.0)))
        for index, (line, color) in enumerate(lines):
            blf.position(font_id, panel_x + 16, panel_top - 28 - index * 21, 0)
            blf.color(font_id, *color)
            blf.draw(font_id, line)

        def graph(label, bottom, series, waveform=False):
            blf.size(font_id, 13)
            blf.position(font_id, panel_x + 16, bottom + 33, 0)
            blf.color(font_id, 0.8, 0.85, 0.89, 1)
            blf.draw(font_id, label)
            for values, color, lane in series:
                if waveform:
                    coords = [(panel_x + 18 + i * 6.25, bottom + 13 + value * 11)
                              for i, value in enumerate(values)]
                    batch = batch_for_shader(shader, "LINE_STRIP", {"pos": coords})
                else:
                    coords = []
                    for i, spike in enumerate(values):
                        if spike:
                            x = panel_x + 18 + i * 6.25
                            coords.extend(((x, bottom + 4 + lane * 12), (x, bottom + 13 + lane * 12)))
                    if not coords:
                        continue
                    batch = batch_for_shader(shader, "LINES", {"pos": coords})
                shader.bind()
                shader.uniform_float("color", color)
                batch.draw(shader)

        graph("FlyWire/Brian2 recorded output  /  SEZ motor replay", panel_top - 494,
              [(flywire_history, (0.95, 0.74, 0.36, 1), 0)], waveform=True)
        graph("Simulated sensor  /  bat echo-neuron spikes", panel_top - 552,
              [(bat_sensors.history["echo"], (0.20, 0.82, 1.0, 1), 0)])
        graph("Simulated sensor  /  moth vision + flame spikes", panel_top - 610,
              [(moth_sensors.history["vision"], (0.40, 0.75, 1.0, 1), 1),
               (moth_sensors.history["flame"], (1.0, 0.62, 0.18, 1), 0)])

    def draw_cinematic_overlay():
        region = bpy.context.region
        if region is None:
            return
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        panel = batch_for_shader(shader, "TRI_FAN", {"pos": [
            (24, region.height - 112), (500, region.height - 112),
            (500, region.height - 24), (24, region.height - 24)]})
        gpu.state.blend_set("ALPHA")
        shader.bind()
        shader.uniform_float("color", (0.015, 0.025, 0.04, 0.72))
        panel.draw(shader)
        gpu.state.blend_set("NONE")
        blf.size(0, 18)
        blf.position(0, 40, region.height - 53, 0)
        blf.color(0, 1.0, 0.82, 0.55, 1)
        blf.draw(0, f"FLYWIRE SIM  •  EPISODE {episode:03d}")
        blf.size(0, 15)
        blf.position(0, 40, region.height - 84, 0)
        blf.color(0, 0.88, 0.92, 0.98, 1)
        blf.draw(0, f"HUNT {max(0, attack_limit - attack_step) / 30:.1f}s  •  CATCH {catches}  •  SURVIVED {escapes}")

    def update_agents(frame):
        nonlocal bat_pos, moth_pos, moth_velocity, bat_velocity, echo_target, attack_step, episode
        nonlocal last_outcome, lamp_visit, catches, escapes, event_timer
        nonlocal bat_decision, moth_decision, feint_side, feint_triggered
        nonlocal last_brain, last_distance, last_flame_distance, tactic_switches, capture_timer, echo_hit_timer, total_ticks
        total_ticks += 1
        last_brain = float(signal[(frame - 1) % len(signal)])
        flywire_history.append(last_brain)
        flicker = 1.0 + 0.12 * math.sin(frame * 0.37) + 0.05 * math.sin(frame * 0.83)
        for index, tongue in enumerate(flames):
            tongue.scale = (1.0 + 0.06 * math.sin(frame * 0.29 + index),
                            1.0 + 0.07 * math.sin(frame * 0.37 + index),
                            flicker * (1.0 + 0.05 * math.sin(frame * 0.43 + index)))
            tongue.rotation_euler[2] = 0.08 * math.sin(frame * 0.21 + index)
            for vertex, original in zip(tongue.data.vertices, flame_shape[index]):
                t = original.z / 0.58
                vertex.co.x = original.x + 0.028 * t * math.sin(frame * 0.32 + t * 8 + index)
                vertex.co.y = original.y + 0.017 * t * math.cos(frame * 0.26 + t * 6 + index)
        flame_light.data.energy = 430.0 + 110.0 * flicker
        if capture_timer > 0:
            capture_dot.scale = (0.10 + 0.05 * math.sin(capture_timer * 0.4),) * 3
            capture_timer -= 1
        else:
            capture_dot.scale = (0, 0, 0)
        if echo_hit_timer > 0:
            echo_hit_dot.scale = (0.075 + 0.035 * math.sin(echo_hit_timer * 1.4),) * 3
            echo_hit_timer -= 1
        else:
            echo_hit_dot.scale = (0, 0, 0)
        sonar_range = SONAR_BASE_RADIUS + SONAR_BRAIN_GAIN * last_brain
        distance = (moth_pos - bat_pos).length
        moth_sensors.step({
            "vision": clamp(1.0 - distance / 3.4, 0.0, 1.0),
            "flame": clamp(1.0 - (moth_pos - lamp_target).length / 3.0, 0.0, 1.0),
        })
        if echo_target is None:
            search = Vector((math.cos(frame * 0.035), math.sin(frame * 0.035), 0.0))
            bat_decision = "ECHO SEARCH"
            bat_aim = bat_pos + search
        else:
            aims = []
            for tactic in bat_learner.current.split("+"):
                aim = echo_target.copy()
                if tactic == "LEAD":
                    aim += moth_velocity * 9.0
                elif tactic == "SWEEP":
                    aim += Vector((0.35 * math.sin(frame * 0.08), 0.35 * math.cos(frame * 0.08), 0.0))
                aims.append(aim)
            bat_aim = sum(aims, Vector()) / len(aims)
            bat_decision = "SONAR → INTERCEPT"
        bat_velocity = (bat_velocity.lerp(direction(bat_aim - bat_pos) * 0.071, 0.24)
                        + boundary_repulsion(bat_pos, arena_half_x, arena_half_y) * 0.10)
        bat_pos += bat_velocity

        sees_bat = distance < 3.4 and sum(list(moth_sensors.history["vision"])[-5:]) > 0
        set_line(moth_sight, moth_pos + Vector((0, 0, 0.06)),
                 bat_pos + Vector((0, 0, 0.06)) if sees_bat else moth_pos + Vector((0, 0, 0.06)))
        set_line(moth_light, moth_pos + Vector((0, 0, 0.04)), lamp_target + Vector((0, 0, 0.04)))
        lamp_delta = lamp_target - moth_pos
        if lamp_visit and lamp_delta.length < 0.95:
            light = direction(Vector((-lamp_delta.y, lamp_delta.x, 0.0)))
        else:
            light = direction(lamp_delta)
        away = direction(moth_pos - bat_pos)
        lateral = Vector((-away.y, away.x, 0.0))
        left = moth_pos + lateral * 0.7
        right = moth_pos - lateral * 0.7
        left_room = min(3.35 - abs(left.x), 2.45 - abs(left.y))
        right_room = min(3.35 - abs(right.x), 2.45 - abs(right.y))
        safe_side = 1.0 if left_room > right_room else -1.0
        def moth_steering(tactic):
            nonlocal feint_side, feint_triggered
            if not sees_bat:
                return light, "SEEK FLAME / SCAN"
            if tactic == "DODGE":
                return (light * 0.55 + (lateral * safe_side + away * 0.35) *
                        (2.4 if distance < 2.5 else 0.4)), "SIDE DODGE"
            if tactic == "FLANK":
                return light * 0.55 + (lateral * safe_side + away * 0.25) * 1.8, "CIRCLE / FLANK"
            if tactic == "LATE":
                return (light * 1.2, "HOLD COURSE") if distance > 1.3 else (
                    lateral * safe_side * 2.6 + away * 0.9, "LAST-MOMENT BREAK")
            if distance < 1.55:
                if not feint_triggered:
                    feint_side *= -1.0
                    feint_triggered = True
                return lateral * feint_side * 2.6 + away * 0.6, "FEINT / REVERSE"
            return light * 0.7 + lateral * feint_side * 0.9, "SHOW FALSE LINE"

        moves = [moth_steering(tactic) for tactic in moth_learner.current.split("+")]
        desired = sum((move for move, _ in moves), Vector()) / len(moves)
        moth_decision = " + ".join(label for _, label in moves)
        moth_velocity = (moth_velocity.lerp(direction(desired) * 0.11, 0.38)
                         + boundary_repulsion(moth_pos, arena_half_x, arena_half_y) * 0.10)
        moth_pos += moth_velocity
        moth_pos.x = clamp(moth_pos.x, -3.35, 3.35)
        moth_pos.y = clamp(moth_pos.y, -2.45, 2.45)
        moth_pos.z = clamp(moth_pos.z, ARENA_FLOOR_Z, ARENA_CEILING_Z)
        if not inside_arena(moth_pos):
            raise AssertionError("Moth left arena")
        bat_pos.x = clamp(bat_pos.x, -arena_half_x, arena_half_x)
        bat_pos.y = clamp(bat_pos.y, -arena_half_y, arena_half_y)
        bat_pos.z = clamp(bat_pos.z, ARENA_FLOOR_Z, ARENA_CEILING_Z)

        if not lamp_visit and (moth_pos - lamp_target).length < 0.42:
            lamp_visit = True
        if event_timer > 0:
            event_timer -= 1
        else:
            last_outcome = "HUNTING"
        bat_arm.location = bat_pos
        if bat_velocity.length > 0:
            old_yaw = bat_arm.rotation_euler[2]
            yaw = math.atan2(-bat_velocity.x, bat_velocity.y)
            turn = math.atan2(math.sin(yaw - old_yaw), math.cos(yaw - old_yaw))
            bat_arm.rotation_euler = (
                math.atan2(bat_velocity.z, max(math.hypot(bat_velocity.x, bat_velocity.y), 1e-6)),
                clamp(-turn * 1.4, -0.32, 0.32), yaw,
            )
        place_moth(moth_pos, moth_velocity)
        if echo_hit_timer > 0:
            # Keep the short ping marker visibly attached to the moving target.
            echo_hit_dot.location = moth_pos
        bpy.context.view_layer.update()
        if bpy.app.background:
            actual_body = sum((moth_body.matrix_world @ Vector(corner) for corner in moth_body.bound_box), Vector()) / 8
            assert (actual_body - moth_pos).length < 0.015, (actual_body, moth_pos)
        mouth = bat_arm.pose.bones.get("Jaw_Lower")
        mouth_world = bat_arm.matrix_world @ mouth.tail if mouth else bat_pos
        pulse_phase = attack_step % SONAR_PULSE_DURATION
        echo_radius = sonar_range * (pulse_phase + 1) / SONAR_PULSE_DURATION
        previous_radius = sonar_range * pulse_phase / SONAR_PULSE_DURATION
        echo_pulse.location = mouth_world
        echo_pulse.scale = (echo_radius,) * 3
        echo_fade.outputs[0].default_value = 0.005 + 0.05 * (1 - echo_radius / sonar_range)
        echo_distance = (moth_pos - mouth_world).length
        forward = direction(bat_velocity)
        in_cone = (SONAR_OMNIDIRECTIONAL or forward.length < 1e-6
                   or forward.dot(direction(moth_pos - mouth_world)) >= math.cos(SONAR_CONE_HALF_ANGLE))
        pulse_hit = in_cone and previous_radius < echo_distance <= echo_radius
        bat_sensors.step({"echo": 1.0 if pulse_hit else 0.0})
        echo_emission.inputs["Color"].default_value = ((1.0, 1.0, 1.0, 1.0) if pulse_hit
                                                          else (0.05, 0.55, 1.0, 1.0))
        if pulse_hit:
            echo_target = moth_pos.copy()
            echo_hit_dot.location = moth_pos
            echo_hit_timer = 5
        attack_step += 1
        new_distance = (moth_pos - bat_pos).length
        new_flame_distance = (moth_pos - lamp_target).length
        bat_reward = 0.35 * (last_distance - new_distance) - 0.002
        moth_reward = 0.35 * (new_distance - last_distance) + 0.06 * (last_flame_distance - new_flame_distance)
        bat_features, moth_features = features()
        last_distance, last_flame_distance = new_distance, new_flame_distance
        caught = mouth_contact(mouth_world, moth_pos)
        animate_particles(total_ticks, moth_pos, flame_world, mouth_world, hit=caught)
        survived = attack_step >= attack_limit
        if caught or survived:
            bat_tactic, moth_tactic = bat_learner.current, moth_learner.current
            duration = attack_step
            bat_learner.step(bat_features, bat_reward + (1.0 if caught else -1.0), terminal=True)
            moth_learner.step(moth_features, moth_reward + (-1.0 if caught else 1.0), terminal=True)
            episode_results.append({
                "episode": episode, "ticks": duration,
                "outcome": "catch" if caught else "survival",
                "bat_tactic": bat_tactic, "moth_tactic": moth_tactic,
                "flame_visited": lamp_visit,
                "bat_spawn": [round(v, 3) for v in spawn_bat],
                "moth_spawn": [round(v, 3) for v in spawn_moth],
            })
        else:
            if bat_learner.step(bat_features, bat_reward):
                tactic_switches += 1
            if moth_learner.step(moth_features, moth_reward):
                tactic_switches += 1
        if caught:
            last_outcome = f"● MOUTH CONTACT  BAT +1 / MOTH -1"
            event_timer = 32
            capture_dot.location = mouth_world
            capture_timer = 32
            episode += 1
            catches += 1
            recent.append("C")
            reset_episode()
        elif survived:
            last_outcome = "SURVIVED  MOTH +1 / BAT -1"
            event_timer = 32
            episode += 1
            escapes += 1
            recent.append("E")
            reset_episode()
        # Persist every live frame: closing Blender resumes this exact hunt.
        if not RESULTS_JSON or caught or survived:
            save_learning()

    scene.frame_set(1)
    def follow_camera():
        center = (bat_pos + moth_pos + lamp_target) / 3.0
        focus = Vector((center.x, center.y, 1.55))
        if CAMERA_PRESET == "wide":
            focus = Vector((0, 0, 1.3))
            desired = focus + Vector((0, -9.0, 6.0))
            target_scale = 10.0
        elif CAMERA_PRESET == "hero":
            desired = bat_pos + Vector((0.0, -3.2, 1.2))
            target_scale = 3.4
        else:
            desired = focus + Vector((0.0, -6.5, 5.7))
            extent = max((pos - center).length for pos in (bat_pos, moth_pos, lamp_target))
            target_scale = clamp(3.5 + extent * 1.55, 5.8, 8.8)
        if bat_velocity.length > SHAKE_SPEED_THRESHOLD:
            shake = Vector((math.sin(total_ticks * 1.7), math.cos(total_ticks * 1.3),
                            math.sin(total_ticks * 2.1))) * 0.012
            desired += shake
        camera.location = camera.location.lerp(desired, 0.10)
        fog_camera_socket.default_value = camera.location
        point_camera(camera, focus)
        camera.data.ortho_scale += (target_scale - camera.data.ortho_scale) * 0.10

    def camera_view():
        for window in bpy.context.window_manager.windows:
            screen = window.screen
            for area in screen.areas:
                if area.type != "VIEW_3D":
                    continue
                area.spaces.active.shading.type = "MATERIAL"
                area.spaces.active.overlay.show_overlays = False
                area.spaces.active.region_3d.view_camera_zoom = 16
                region = next((r for r in area.regions if r.type == "WINDOW"), None)
                if region:
                    with bpy.context.temp_override(window=window, screen=screen, area=area, region=region):
                        bpy.ops.view3d.view_camera()
        return None

    if not bpy.app.background:
        overlay = draw_cinematic_overlay if HUD_MODE == "cinematic" else draw_decision_feed
        bpy.types.SpaceView3D.draw_handler_add(overlay, (), "WINDOW", "POST_PIXEL")
        bpy.app.timers.register(camera_view, first_interval=1.0)
    print("Predator–prey tactics active; scores saved after every encounter")
    if not bpy.app.background:
        def advance():
            frame = scene.frame_current
            update_agents(frame)
            follow_camera()
            if frame % 24 == 0:
                print(
                    f"episode={episode:03d} tick={attack_step:03d}/{attack_limit} "
                    f"bat={bat_learner.current}:{bat_decision} "
                    f"moth={moth_learner.current}:{moth_decision} "
                    f"range={(moth_pos - bat_pos).length:.2f} "
                    f"flame={(moth_pos - lamp_target).length:.2f} "
                    f"catches={catches} escapes={escapes} outcome={last_outcome}"
                )
            scene.frame_set(frame + 1 if frame < scene.frame_end else scene.frame_start)
            return 1.0 / scene.render.fps

        bpy.app.timers.register(advance, first_interval=0.0)
    update_agents.save = save_learning
    return update_agents


if bpy.app.background:
    assert mouth_contact(Vector((0, 0, 1.65)), Vector((0.1, 0, 1.65)))
    assert not mouth_contact(Vector((0, 0, 1.65)), Vector((0.3, 0, 1.65)))
    assert not mouth_contact(Vector((0, 0, 2.8)), Vector((0, 0, 1.65)))
    assert inside_arena(Vector((3.35, 2.45, 1.65)))
    assert not inside_arena(Vector((3.36, 2.45, 1.65)))
update = setup()
bpy.app.driver_namespace["flywiresim_update"] = update
if bpy.app.background:
    for tick in range(int(os.environ.get("FLYWIRE_TEST_STEPS", "0"))):
        bpy.context.scene.frame_set(1 + tick % bpy.context.scene.frame_end)
        update(tick)
    if os.environ.get("FLYWIRE_TEST_STEPS"):
        update.save()
        print("TEST_RESULT", LEARNING_JSON.read_text() if LEARNING_JSON.exists() else "no episodes")
    render_path = os.environ.get("FLYWIRE_RENDER_PATH")
    if render_path:
        if os.environ.get("FLYWIRE_RENDER_EXPOSURE"):
            bpy.context.scene.view_settings.exposure = float(os.environ["FLYWIRE_RENDER_EXPOSURE"])
        render_tick = int(os.environ.get("FLYWIRE_RENDER_TICK", "48"))
        bpy.context.scene.frame_set(1 + render_tick % bpy.context.scene.frame_end)
        update(render_tick)
        bpy.context.scene.render.filepath = str(Path(render_path).resolve())
        bpy.ops.render.render(write_still=True)
        print(f"RENDER_RESULT {bpy.context.scene.render.filepath}")
    render_dir = os.environ.get("FLYWIRE_RENDER_SEQUENCE_DIR")
    if render_dir:
        output_dir = Path(render_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        if os.environ.get("FLYWIRE_RENDER_EXPOSURE"):
            bpy.context.scene.view_settings.exposure = float(os.environ["FLYWIRE_RENDER_EXPOSURE"])
        count = int(os.environ.get("FLYWIRE_RENDER_SEQUENCE_FRAMES", "600"))
        source_step = max(1, int(os.environ.get("FLYWIRE_RENDER_SEQUENCE_SOURCE_STEP", "3")))
        bpy.context.scene.render.resolution_x = int(os.environ.get("FLYWIRE_RENDER_WIDTH", "480"))
        bpy.context.scene.render.resolution_y = int(os.environ.get("FLYWIRE_RENDER_HEIGHT", "270"))
        bpy.context.scene.render.resolution_percentage = 100
        samples = int(os.environ.get("FLYWIRE_RENDER_SAMPLES", "8"))
        if hasattr(bpy.context.scene, "eevee"):
            for property_name in ("taa_render_samples", "taa_samples"):
                if hasattr(bpy.context.scene.eevee, property_name):
                    setattr(bpy.context.scene.eevee, property_name, samples)
        for output_index in range(count):
            for _ in range(source_step):
                update(output_index)
            bpy.context.scene.frame_set(1 + output_index % bpy.context.scene.frame_end)
            bpy.context.scene.render.filepath = str(output_dir / f"frame_{output_index:05d}.png")
            bpy.ops.render.render(write_still=True)
        print(f"RENDER_SEQUENCE_RESULT {output_dir} frames={count}")
