import re
import unicodedata
import bmesh
import bpy
import json
import os
import importlib.util
from math import radians
from mathutils import Matrix, Vector

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # i guess we're doubling up

import sys

sys.path.append(SCRIPT_DIR)

from ttyd_agb_structs import *

MODEL_NAME = 'FRY_mekri'
TEX_DIR = MODEL_NAME + '_tex'

armature = bpy.data.objects.new('Armature', bpy.data.armatures.new('Armature'))

materials = {}
def get_simple_mat_for_tex(texture_path):
    if texture_path in materials:
        return materials[texture_path]

    new_material = bpy.data.materials.new('mat')
    new_material.use_nodes = True
    bsdf = new_material.node_tree.nodes['Principled BSDF']
    # bsdf.inputs['Specular'].default_value = 0
    tex_image = new_material.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(texture_path)
    new_material.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])
    new_material.node_tree.links.new(bsdf.inputs['Alpha'], tex_image.outputs['Alpha'])
    new_material.surface_render_method = 'BLENDED'

    materials[texture_path] = new_material
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
objects = []
scale_matrices = {}
deferred_parenting = []
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
            assert(subshape.sampler_count == 1)

            sampler = agb.samplers[subshape.sampler_indices[0]]
            texture = agb.textures[sampler.texture_base_id]

            new_material = get_simple_mat_for_tex(os.path.join(TEX_DIR, f'{MODEL_NAME}--{texture.tpl_index}.png'))
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

    # if cur_object.parent and cur_object.parent.type == 'MESH':
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

    if has_shape:
        deferred_parenting.append((cur_object, bone.name))
    else:
        bpy.data.objects.remove(cur_object, do_unlink=True)
        
    #endregion

    objects.append(bone)

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

    for cur_object, bone_name in deferred_parenting:
        matrix_world = cur_object.matrix_world.copy()

        cur_object.parent = armature
        cur_object.parent_type = 'BONE'
        cur_object.parent_bone = bone_name

        bone_matrix_world = armature.matrix_world @ armature.pose.bones[bone_name].matrix
        cur_object.matrix_basis = bone_matrix_world.inverted() @ matrix_world

        objects.append(cur_object)
    #endregion

    agb.anims = agb.anims

    scene = bpy.context.scene

    for anim in agb.anims:
        data = anim.data
        if not data:
            continue

        anim_name = "!" + anim.name
        action = bpy.data.actions.new(anim_name)
        action.use_fake_user = True

        for bone in armature.pose.bones:
            for property in ['location', 'rotation_quaternion', 'scale']:
                prop_data = getattr(bone, property, None)
                if prop_data is None:
                    continue
                
                for i in range(4 if property == 'rotation_quaternion' else 3):
                    fcurve = action.fcurves.new(data_path='pose.bones["{}"].{}'.format(bone.name, property), index=i)
                    if len(prop_data) > i:
                        fcurve.keyframe_points.insert(1, prop_data[i])
                        fcurve.keyframe_points.insert(5, prop_data[i])
        
        if not armature.animation_data:
           armature.animation_data_create()
        
        track = armature.animation_data.nla_tracks.new()
        track.name = anim_name
        track.mute = True

        strip = track.strips.new(name=anim_name, start=1, action=action)
        strip.name = anim_name
        
    print("\n\n-- FINISHED IMPORT --\n\n")