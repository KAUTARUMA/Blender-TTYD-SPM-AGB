from collections import defaultdict
import math
import re
import unicodedata
import bmesh
import bpy
import json
import os
import importlib.util
from math import radians
from mathutils import Matrix, Vector

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.append(SCRIPT_DIR)

from ttyd_agb_structs import *

MODEL_NAME = 'FRY_dash'
TEX_DIR = MODEL_NAME + '_tex'

armature = bpy.data.objects.new('Armature', bpy.data.armatures.new('Armature'))

materials = {}
texture_sequence = []

def get_simple_mat_for_tex(texture_id, sampler_index):
    if texture_id in materials:
        return materials[texture_id][0]

    new_material = bpy.data.materials.new('mat')
    new_material.name = f'{MODEL_NAME}_{texture_id}'
    new_material.use_nodes = True
    bsdf = new_material.node_tree.nodes['Principled BSDF']

    tex_image = new_material.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(texture_sequence[0])
    tex_image.image.source = 'SEQUENCE'

    tex_image.image_user.frame_start = 1
    tex_image.image_user.frame_duration = 1
    tex_image.image_user.frame_offset = texture_id - 1
    tex_image.image_user.use_auto_refresh = True

    new_material.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
    new_material.node_tree.links.new(bsdf.inputs['Alpha'], tex_image.outputs['Alpha'])
    new_material.surface_render_method = 'BLENDED'

    materials[texture_id] = (new_material, sampler_index)
    return new_material


def group_transform_to_matrices(gt, joint_parent_gt=None):
    translation = Matrix.Translation(Vector((gt.translation.x, gt.translation.y, gt.translation.z)))

    scale_x = Matrix.Scale(gt.scale.x, 4, Vector((1, 0, 0)))
    scale_y = Matrix.Scale(gt.scale.y, 4, Vector((0, 1, 0)))
    scale_z = Matrix.Scale(gt.scale.z, 4, Vector((0, 0, 1)))
    scale = scale_x @ scale_y @ scale_z

    rot2deg_x = Matrix.Rotation(radians(gt.rotation_in_2deg.x * 2), 4, 'X')
    rot2deg_y = Matrix.Rotation(radians(gt.rotation_in_2deg.y * 2), 4, 'Y')
    rot2deg_z = Matrix.Rotation(radians(gt.rotation_in_2deg.z * 2), 4, 'Z')
    rot2deg = rot2deg_z @ rot2deg_y @ rot2deg_x

    joint_rot_x = Matrix.Rotation(radians(gt.joint_post_rotation_in_deg.x), 4, 'X')
    joint_rot_y = Matrix.Rotation(radians(gt.joint_post_rotation_in_deg.y), 4, 'Y')
    joint_rot_z = Matrix.Rotation(radians(gt.joint_post_rotation_in_deg.z), 4, 'Z')
    joint_rot = joint_rot_z @ joint_rot_y @ joint_rot_x

    scale = Matrix.Translation(
        Vector((gt.transform_scale_pivot.x + gt.transform_scale_offset.x,
                gt.transform_scale_pivot.y + gt.transform_scale_offset.y,
                gt.transform_scale_pivot.z + gt.transform_scale_offset.z))
    ) @ scale
    scale = scale @ Matrix.Translation(Vector((-gt.transform_scale_pivot.x, -gt.transform_scale_pivot.y, -gt.transform_scale_pivot.z)))

    rotation = joint_rot @ rot2deg
    rotation = Matrix.Translation(
        Vector((gt.transform_rotation_pivot.x + gt.transform_rotation_offset.x,
                gt.transform_rotation_pivot.y + gt.transform_rotation_offset.y,
                gt.transform_rotation_pivot.z + gt.transform_rotation_offset.z))
    ) @ rotation
    rotation = rotation @ Matrix.Translation(Vector((-gt.transform_rotation_pivot.x, -gt.transform_rotation_pivot.y, -gt.transform_rotation_pivot.z)))

    if joint_parent_gt:
        parent_scale_x = Matrix.Scale(1 / joint_parent_gt.scale.x, 4, Vector((1, 0, 0)))
        parent_scale_y = Matrix.Scale(1 / joint_parent_gt.scale.y, 4, Vector((0, 1, 0)))
        parent_scale_z = Matrix.Scale(1 / joint_parent_gt.scale.z, 4, Vector((0, 0, 1)))
        parent_scale = parent_scale_x @ parent_scale_y @ parent_scale_z
        rotation = parent_scale @ rotation

    return translation, rotation, scale


# note: this only supports one sampler
agb_data = {}
scale_matrices = {}

def group_to_object(group, collection, parent=None, parent_group=None): # parent is (group, obj)
    has_shape = group.shape_id > -1

    if has_shape:
        shape = agb.shapes[group.shape_id]

        shape_mesh = bpy.data.meshes.new(shape.name)

        bm = bmesh.new()
        uv_layer = bm.loops.layers.uv.new()
        color_layer = bm.loops.layers.color.new('color')

        assert(shape.subshape_count == 1)

        prebuilt_shape_vertices = []
        for v in agb.vertex_positions[shape.vertex_position_data_base_index:shape.vertex_position_data_base_index + shape.vertex_position_data_count]:
            prebuilt_shape_vertices.append(bm.verts.new((v.x, v.y, v.z)))

        shape_vert_normals = [None] * len(prebuilt_shape_vertices)
        for subshape in agb.subshapes[shape.subshape_base_index:shape.subshape_base_index + shape.subshape_count]:
            
            # i have no clue if im doing this right
            for sampler_index in subshape.sampler_indices:
                if sampler_index == -1: continue

                sampler = agb.samplers[sampler_index]
                texture = agb.textures[sampler.texture_base_id]

                new_material = get_simple_mat_for_tex(texture.tpl_index, sampler_index)
                shape_mesh.materials.append(new_material)

            for polygon in agb.polygons[subshape.polygon_base_index:subshape.polygon_base_index + subshape.polygon_count]:
                assert(polygon.vertex_count >= 3)

                polygon_verts = []
                vert_uvs = []
                vert_colors = []
                for vtx_ix in range(polygon.vertex_count):
                    abs_vertex_index = polygon.vertex_base_index + vtx_ix

                    uv = agb.vertex_texture_coordinates[shape.vertex_texture_coordinate0_data_base_index + agb.vertex_texture_coordinate_indices[subshape.vertex_texture_coordinate_indices_base_index[0] + abs_vertex_index]]
                    color = agb.vertex_colors[shape.vertex_color_data_base_index + agb.vertex_color_indices[subshape.vertex_color_base_indices_base_index + abs_vertex_index]]
                    normal = agb.vertex_normals[shape.vertex_normal_data_base_index + agb.vertex_normal_indices[subshape.vertex_normal_base_indices_base_index + abs_vertex_index]]

                    shape_pos_vert_ix = agb.vertex_position_indices[subshape.vertex_position_indices_base_index + abs_vertex_index]
                    polygon_verts.append(prebuilt_shape_vertices[shape_pos_vert_ix])
                    shape_vert_normals[shape_pos_vert_ix] = (normal.x, normal.y, normal.z)

                    vert_uvs.append((uv.x, 1 - uv.y))
                    vert_colors.append((color.r / 255, color.g / 255, color.b / 255, color.a / 255))

                new_face = bm.faces.new(polygon_verts)
                for lix, loop in enumerate(new_face.loops):
                    loop[uv_layer].uv = vert_uvs[lix]
                    loop[color_layer] = vert_colors[lix]

        bm.to_mesh(shape_mesh)
        bm.free()

        shape_mesh.validate(clean_customdata=False)
        shape_mesh.update()
        shape_mesh.normals_split_custom_set_from_vertices(shape_vert_normals)

    cur_object = bpy.data.objects.new(group.name, shape_mesh if has_shape else None)
    
    collection.objects.link(cur_object)

    gt = agb.group_transform_data[group.corrected_transform_index]
    parent_gt = None
    if parent_group:
        parent_gt = agb.group_transform_data[parent_group.corrected_transform_index]

    translation, rotation, scale = group_transform_to_matrices(gt, parent_gt if group.is_joint else None)
    scale_matrices[cur_object] = scale
    transform = translation @ rotation @ scale

    rot_pivot = gt.transform_rotation_pivot
    origin_offset = Matrix.Translation(Vector((-rot_pivot.x / gt.scale.x, -rot_pivot.y / gt.scale.y, -rot_pivot.z / gt.scale.z)))
    if has_shape:
        transform = transform @ origin_offset.inverted()
        cur_object.data.transform(origin_offset)

    # if cur_object.parent:
    #     parent_pivot = parent_gt.transform_rotation_pivot
    #     parent_scale = scale_matrices[cur_object.parent]
    #     transform = parent_scale.inverted() @ Matrix.Translation(Vector((-parent_pivot.x, -parent_pivot.y, -parent_pivot.z))) @ parent_scale @ transform

    cur_object.matrix_local = transform

    #region dumb bone hack
    bone = armature.data.edit_bones.new(cur_object.name)
    bone.head = cur_object.location
    bone.tail = cur_object.location + Vector((0, 1, 0))

    if parent:
        bone.parent = parent
        bone.use_connect = False

    if not has_shape:
        bpy.data.objects.remove(cur_object, do_unlink=True)
        
    #endregion

    agb_data[bone.name] = {"group": group, "object": cur_object if has_shape else None}

    if group.child_group_id > -1:
        group_to_object(agb.groups[group.child_group_id], collection, bone, group)
    if group.next_group_id > -1:
        group_to_object(agb.groups[group.next_group_id], collection, parent)

if __name__ == "__main__":
    print("\n\n-- RUNNING IMPORT --\n\n")

    MODEL_FILE = os.path.join(SCRIPT_DIR, MODEL_NAME)
    TEX_DIR = os.path.join(SCRIPT_DIR, TEX_DIR)
    
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file does not exist: {repr(MODEL_FILE)}")
    
    if not os.path.exists(TEX_DIR):
        raise FileNotFoundError(f"Texture directory does not exist: {repr(TEX_DIR)}")
    
    for file in sorted(os.listdir(TEX_DIR)):
        if file.startswith(f"{MODEL_NAME}--"):
            texture_sequence.append(os.path.join(TEX_DIR, file))
    
    with open(MODEL_FILE, 'rb') as f:
        agb = agb_read(f)

    new_collection = bpy.data.collections.new(agb.header.model_name)
    bpy.context.scene.collection.children.link(new_collection)

    armature.name = agb.header.model_name
    new_collection.objects.link(armature)

    root_group = agb.groups[-1]

    #region object parsing
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode='EDIT')

    group_to_object(root_group, new_collection)

    bpy.ops.object.mode_set(mode='OBJECT')

    for bone_name in agb_data.keys():
        bone_data = agb_data[bone_name]
        cur_object = bone_data["object"]
        if cur_object is None: continue

        matrix_world = cur_object.matrix_world.copy()

        cur_object.parent = armature
        cur_object.parent_type = 'BONE'
        cur_object.parent_bone = bone_name

        bone_matrix_world = armature.matrix_world @ armature.pose.bones[bone_name].matrix
        cur_object.matrix_basis = bone_matrix_world.inverted() @ matrix_world
    #endregion

    scene = bpy.context.scene

    if not armature.animation_data:
        armature.animation_data_create()
    
    scene.render.fps = 60
    # animation parsing
    for anim in agb.anims:
        data = anim.data
        if not data:
            continue
        
        # why the hell is base info a table lol
        anim_name = f'!' + ('@' if data.base_info[0].loop == 1 else '') + f'{anim.name}'
        action = bpy.data.actions.new(anim_name)
        # action.use_fake_user = True

        #region group visibility parsing
        frames_visibility = defaultdict(dict)
        current_visibility = {
            vg_id: agb.visibility_groups[vg_id]
            for vg_id in range(len(agb.visibility_groups))
        }

        for keyframe in data.keyframes:
            visgroup_id = 0
            for i in range(keyframe.visibility_group_delta_count):
                delta = data.visibility_group_deltas[keyframe.visibility_group_delta_base_index + i]
                visgroup_id += delta.index_delta

                if delta.visible == 0:
                    continue
                elif delta.visible == 1:
                    current_visibility[visgroup_id] = 1
                elif delta.visible == -1:
                    current_visibility[visgroup_id] = 0

                frames_visibility[keyframe.time][visgroup_id] = current_visibility[visgroup_id]
        #regionend

        #region material animation parsing
        current_material_values = {}
        last_mat_values = {}

        for keyframe in data.keyframes:
            for i in range(keyframe.texture_coordinate_transform_delta_count):
                delta = data.texture_coordinate_transform_deltas[keyframe.texture_coordinate_transform_delta_base_index + i]

                material = next((materials[n][0] for n in materials if materials[n][1] == delta.index_delta), None)
                if not material:
                    continue

                tex_node = next((n for n in material.node_tree.nodes if n.type == 'TEX_IMAGE'), None)
                if not tex_node:
                    continue
                
                last_value = last_mat_values.get(delta.index_delta, tex_node.image_user.frame_offset + 1)

                last_value += delta.frame_ext_delta
                last_mat_values[delta.index_delta] = last_value

                if not current_material_values.get(keyframe.time):
                    current_material_values[keyframe.time] = {}

                current_material_values[keyframe.time] = (delta.index_delta, last_value)
        #regionend

        for bone in armature.pose.bones:
            if bone.name not in agb_data: continue

            group = agb_data[bone.name]["group"]

            group_id = group.visibility_group_id
            base_id = group.transform_base_index

            #region group animations
            bone.rotation_mode = 'XYZ'

            channel_indices = {
                0: ("location", 0),
                1: ("location", 1),
                2: ("location", 2),
                3: ("scale", 0),
                4: ("scale", 1),
                5: ("scale", 2),
                6: ("rotation_euler", 0),  # in 2deg
                7: ("rotation_euler", 1),
                8: ("rotation_euler", 2),

                # additional channels that i have no idea what to do with
                9: ("joint_post_rotation", 0),
                10: ("joint_post_rotation", 1),
                11: ("joint_post_rotation", 2),
                12: ("rotation_pivot", 0),
                13: ("rotation_pivot", 1),
                14: ("rotation_pivot", 2),
                15: ("scale_pivot", 0),
                16: ("scale_pivot", 1),
                17: ("scale_pivot", 2),
                18: ("rotation_offset", 0),
                19: ("rotation_offset", 1),
                20: ("rotation_offset", 2),
                21: ("scale_offset", 0),
                22: ("scale_offset", 1),
                23: ("scale_offset", 2),
            }

            # reconstruct absolute values per transform offset
            frame_values = {}
            current_values = {}

            for keyframe in data.keyframes:
                frame = keyframe.time
                frame_values[frame] = {}

                delta_base = keyframe.group_transform_data_delta_base_index
                delta_count = keyframe.group_transform_data_delta_count

                rel_index = 0
                transform_offset = 0

                for delta in data.group_transform_data_deltas[delta_base : delta_base + delta_count]:
                    rel_index = delta.index_delta
                    transform_offset += rel_index

                    value = delta.value_delta / 16.0
                    abs_index = transform_offset

                    channel_index = abs_index - base_id
                    if channel_index >= 3 and channel_index <= 5: # if scale
                        prev_value = current_values.get(abs_index, 1.0)
                    else:
                        prev_value = current_values.get(abs_index, 0.0)
                    
                    new_value = prev_value + value
                    current_values[abs_index] = new_value

                    frame_values[frame][abs_index] = (new_value, delta.tangent_in_deg, delta.tangent_out_deg)

                # fill in any values not modified this frame with previous values
                # commented out because i think this causes unintended behavior with the easing but idk
                # for i in range(base_id, base_id + len(channel_indices)):
                #     if i not in frame_values[frame]:
                #         if i in current_values:
                #             frame_values[frame][i] = (current_values[i], None, None)
            
            # Apply keyframes to fcurves
            for i in range(len(channel_indices)):
                abs_index = base_id + i

                if i > 8: continue

                data_path, axis = channel_indices[i]

                fcurve = action.fcurves.new(data_path=f'pose.bones["{bone.name}"].{data_path}', index=axis)
                fcurve.keyframe_points.insert(0, getattr(bone, data_path)[axis])
                
                for frame, values in frame_values.items():
                    if abs_index in values:
                        delta_info = values[abs_index]
                        value = delta_info[0]

                        if i >= 6 and i <= 8:
                            value = value * (math.pi/180)

                        key = fcurve.keyframe_points.insert(frame, value)

                        if delta_info[1] != None and delta_info[2] != None:
                            continue # broken atm
                            tangent_in = math.tan(math.radians(delta_info[1])) if delta_info[1] is not None else 0.0
                            tangent_out = math.tan(math.radians(delta_info[2])) if delta_info[2] is not None else 0.0
                            key.handle_left_type = 'FREE'
                            key.handle_right_type = 'FREE'

                            delta = 1.0
                            key.handle_left.y = key.co.y - tangent_in * delta
                            key.handle_left.x = key.co.x - delta
                            key.handle_right.y = key.co.y + tangent_out * delta
                            key.handle_right.x = key.co.x + delta
            #regionend

            #region object visibility actions    
            object = agb_data[bone.name]["object"]
            if object is None:
                continue

            if not object.animation_data:
                object.animation_data_create()
            
            action_name = f"{anim.name}_{object.name}_vis"
            obj_action = bpy.data.actions.new(name=action_name)
            object.animation_data.action = obj_action
                
            visibility_action = object.animation_data.action

            fcurve = visibility_action.fcurves.new(data_path="hide_viewport")

            for frame in sorted(frames_visibility.keys()):
                visibility_at_frame = frames_visibility[frame].get(group_id)
                if visibility_at_frame is not None:
                    hidden = not visibility_at_frame
                    fcurve.keyframe_points.insert(frame, hidden)
            
            obj_track = object.animation_data.nla_tracks.new()
            obj_track.name = f'{anim.name}_{object.name}_vis'
            obj_track.mute = True
            
            strip = obj_track.strips.new(name=action_name, start=1, action=obj_action)
            strip.name = action_name
            #regionend

        #region material tex actions
        for tex_id in materials:
            material_tuple = materials[tex_id]
            material = material_tuple[0]

            tex_node = next((n for n in material.node_tree.nodes if n.type == 'TEX_IMAGE'), None)
            if not tex_node:
                continue

            if not material.animation_data:
                material.animation_data_create()

            data_path = f'node_tree.nodes["{tex_node.name}"].image_user.frame_offset'

            action_name = f"{anim.name}_{material.name}_tex"

            mat_action = bpy.data.actions.new(name=action_name)
            material.animation_data.action = mat_action

            fcurve = mat_action.fcurves.find(data_path)

            if not fcurve:
                fcurve = mat_action.fcurves.new(data_path=data_path)

            for frame in sorted(current_material_values.keys()):
                material_values = current_material_values[frame]
                if material_values[0] != material_tuple[1]: continue

                key = fcurve.keyframe_points.insert(frame, material_values[1] - 1)
                key.interpolation = 'CONSTANT'
            
            mat_track = material.animation_data.nla_tracks.new()
            mat_track.name = action_name
            mat_track.mute = True

            strip = mat_track.strips.new(name=mat_action.name, start=1, action=mat_action)
        #regionend
        
        track = armature.animation_data.nla_tracks.new()
        track.name = anim_name
        track.mute = True
        strip = track.strips.new(name=action.name, start=1, action=action)
    
    bpy.context.scene.frame_set(bpy.context.scene.frame_current)
        
    print("\n\n-- FINISHED IMPORT --\n\n")