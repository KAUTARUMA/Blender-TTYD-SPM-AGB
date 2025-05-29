import bpy
import math
import os
from collections import defaultdict
from copy import copy
from datetime import datetime
from decimal import Decimal
from math import degrees
from mathutils import Vector

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # i guess we're doubling up

# force reload changes (prevent blender module caching)
import sys
sys.path.append(SCRIPT_DIR)
from ttyd_agb_structs import *

# name of the collection to export
MODEL_NAME = 'FRY_dash'

# path to save the exported model to
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'exports', f'{MODEL_NAME}')

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


# what it says on the tin
DISABLE_VERTEX_COLORS = False
DISABLE_ANIMATED_NORMALS = False

# this upscales your model to allow for more precision in animations.
# you'll probably need it if your model has complex geometry.
# causes major bloat (although the growth is ~logarithmic)
SUPERRESOLUTION_FACTOR = 2

# if your translate or scale animations are choppy,
# you'll want to increase this. note that model
# superresolution already affects translation.
TRANSFORM_SUPERRESOLUTION_FACTOR = 2


ZERO_VEC3 = Vec3(0, 0, 0)
ZERO_VECTOR = Vector()
ZERO_LIST3 = [0, 0, 0]

armature = None

def is_nan(x):
    return Decimal(x).is_nan()

def agbvec(vec):
    if len(vec) == 3:
        return Vec3(vec.x, vec.y, vec.z)
    elif len(vec) == 2:
        return Vec2(vec.x, vec.y)
    else:
        raise ValueError(f"vec len is neither 3 or 2: {vec}")

def color4f(r, g, b, a):
    r = int(255 * r)
    g = int(255 * g)
    b = int(255 * b)
    a = int(255 * a)
    return Color4(r, g, b, a)

COLOR_GRAY = color4f(0.8, 0.8, 0.8, 1.0)

def get_tsrf(obj):
    if has_transform_anim_map.get(obj, False):
        return TRANSFORM_SUPERRESOLUTION_FACTOR
    return 1

def get_bone_shape(bone):
    return next((child for child in armature.children if child.name == bone.name), None)

# unfinished and suboptimal/sloppy
def material_bullshit(obj):
    noret = (None, None, None, None, None)

    materials = [mat for mat in obj.data.materials if mat]
    if not materials:
        return noret

    assert(len(materials) == 1)

    mat = materials[0]
    assert(mat.use_nodes)

    out_nodes = [node for node in mat.node_tree.nodes if node.type == 'OUTPUT_MATERIAL']
    assert(len(out_nodes) == 1)
    out_node = out_nodes[0]
    assert(not out_node.inputs['Volume'].is_linked)
    assert(not out_node.inputs['Displacement'].is_linked)

    bsdf = out_node.inputs['Surface']
    if not bsdf.is_linked:
        return noret
    bsdf = bsdf.links[0].from_node
    # assert(bsdf.type == 'BSDF_DIFFUSE')
    assert(not bsdf.inputs['Roughness'].is_linked)
    assert(not bsdf.inputs['Normal'].is_linked)

    color = bsdf.inputs['Base Color']
    if not color.is_linked:
        return noret
    
    color_sock = color.links[0].from_socket
    color_node = color_sock.node

    vtxcol = None
    texture = None
    if color_node.type == 'MIX_RGB':
        assert(color_node.blend_type == 'MULTIPLY')
        assert(color_node.inputs['Fac'].default_value == 1)

        for input_name in ('Color1', 'Color2'):
            input = color_node.inputs[input_name]
            if input.is_linked:
                input = input.links[0].from_socket
                if input.node.type == 'VERTEX_COLOR' and not vtxcol:
                    vtxcol = input
                elif input.node.type == 'TEX_IMAGE' and not texture:
                    texture = input
                else:
                    assert(False)
    else:
        if color_node.type == 'VERTEX_COLOR':
            vtxcol = color_sock
        elif color_node.type == 'TEX_IMAGE':
            texture = color_sock
        else:
            assert(False)

    color_layer = None
    if vtxcol:
        assert(vtxcol.name == 'Color')
        layer_name = vtxcol.node.layer_name
        if layer_name:
            color_layer = obj.data.vertex_colors[layer_name].data

    tex_img = None
    ext_mode = None
    uv_layer = None
    if texture:
        assert(texture.name == 'Color')
        tex_node = texture.node

        tex_img = tex_node.image
        assert(tex_img)

        ext_mode = tex_node.extension
        assert(ext_mode in ['EXTEND', 'REPEAT'])

        uv_input = tex_node.inputs['Vector']

        if uv_input.is_linked:
            uv_layer = uv_input.links[0].from_node.uv_map
        else:
            uv_layer = obj.data.uv_layers.active.name

        assert(uv_layer)
        uv_layer = obj.data.uv_layers[uv_layer].data

    return (mat, color_layer, tex_img, ext_mode, uv_layer)

# this used to actually process textures to tpl,
# but it turned out to be too slow, so for now it's just a counter
def add_texture(image, ext_mode):
    #if image in tpl_image_map:
    #    return tpl_image_map[image]

    #tex_count_before = len(tpl_image_map)
    #tpl_image_map[image] = tex_count_before
    #return tex_count_before

    print(int(image.name.split('--')[1].split('.')[0]))
    return int(image.name.split('--')[1].split('.')[0])


def object_to_group(bone, ident=0):
    depsgraph = bpy.context.evaluated_depsgraph_get()

    shape = get_bone_shape(bone)
    has_shape = shape is not None

    shape_eval = None

    if has_shape:
        shape_eval = shape.evaluated_get(depsgraph)
    else:
        shape_eval = armature.evaluated_get(depsgraph).pose.bones.get(bone.name)

    if has_shape:
        mesh_base_counts[shape] = (len(shape_eval.data.vertices), len(shape_eval.data.polygons), len(shape_eval.data.loops))

        vtx_count_before = len(agb.vertex_positions)
        for vertex in shape_eval.data.vertices.values():
            vtxid = len(agb.vertex_positions)
            vtxpath2id[repr(vertex)] = vtxid

            co_superres = vertex.co * SUPERRESOLUTION_FACTOR
            vtxid_base_attrs[vtxid] = dict(position=co_superres, normal=copy(vertex.normal))
            agb.vertex_positions.append(agbvec(co_superres))
            agb.vertex_normals.append(agbvec(vertex.normal))

            # find base bbox
            vtxco_world = shape_eval.matrix_world @ vertex.co
            for ix in range(3):
                co = vtxco_world[ix]
                agb.header.bbox_min[ix] = min(agb.header.bbox_min[ix], co)
                agb.header.bbox_max[ix] = max(agb.header.bbox_max[ix], co)

        mat, colors, tex_img, ext_mode, uvs = material_bullshit(shape)

        backface_culling = False
        if mat:
            backface_culling = mat.use_backface_culling

        vtxclr_count_before = len(agb.vertex_colors)
        colors = (not DISABLE_VERTEX_COLORS) and colors
        if not colors:
            agb.vertex_colors.append(COLOR_GRAY)

        uvs = uvs
        if not uvs:
            agb.vertex_texture_coordinates.append(Vec2(0, 0))

        visited_vertices = [False] * len(shape_eval.data.vertices)
        adjusted_loop_ix = 0
        vtx_loop_ix_map_iguess = {}
        vtxtexcoord_count_before = len(agb.vertex_texture_coordinates)
        for lix, loop in enumerate(shape_eval.data.loops):
            if visited_vertices[loop.vertex_index]:
                continue

            if colors and lix < len(colors):
                # this may be wrong, i'm not familiar with colorspace issues
                # and i don't need vertex colors enough to want to look into this
                agb.vertex_colors.append(color4f(*colors[lix].color))

            if uvs and len(uvs) > lix:
                uv = uvs[lix].uv
                agb.vertex_texture_coordinates.append(Vec2(uv.x, 1 - uv.y))

            vtx_loop_ix_map_iguess[loop.vertex_index] = adjusted_loop_ix

            visited_vertices[loop.vertex_index] = True
            adjusted_loop_ix += 1

        polygon_count_before = len(agb.polygons)
        vtxpos_indices_count_before = len(agb.vertex_position_indices)
        vtxnrm_indices_count_before = len(agb.vertex_normal_indices)
        vtxclr_indices_count_before = len(agb.vertex_color_indices)
        vtxtexcoord_indices_count_before = len(agb.vertex_texture_coordinate_indices)
        for polygon in shape_eval.data.polygons.values():
            vertex_count = len(polygon.vertices)
            agb.polygons.append(AGBPolygon(
                vertex_base_index=len(agb.vertex_position_indices) - vtxpos_indices_count_before,
                vertex_count=vertex_count,
            ))

            for vertex_ix in polygon.vertices:
                agb.vertex_position_indices.append(vertex_ix)
                agb.vertex_normal_indices.append(vertex_ix)
                agb.vertex_color_indices.append(vtx_loop_ix_map_iguess[vertex_ix] if colors else 0)
                agb.vertex_texture_coordinate_indices.append(vtx_loop_ix_map_iguess[vertex_ix] if uvs else 0)

        if tex_img:
            tex_id = add_texture(tex_img, ext_mode)

            texture = AGBTexture(
                unk_0=0, # ?
                tpl_index=tex_id,
                wbUnused=0, # ?
                unk_c="Hello to you too, Diagamma! :D",
                unk_38=[0, 0], # ????
            )
            texture_count_before = len(agb.textures)
            agb.textures.append(texture)

            texcoord_transform = AGBTextureCoordinateTransform(
                texture_frame_offset=0,
                translation_x=0,
                translation_y=0,
                scale_x=1,
                scale_y=1,
                rotation=0,
            )
            texcoord_transform_count_before = len(agb.texture_coordinate_transforms)
            agb.texture_coordinate_transforms.append(texcoord_transform)

            sampler = AGBSampler(
                # this used to be the texture thing index, but it seems to be unused?
                # it looks like the tpl index is directly lifted from texture_base_id
                texture_base_id=texture_count_before,
                wrap_flags=0x80000000, # ig?
            )
            sampler_count_before = len(agb.samplers)
            agb.samplers.append(sampler)
        else: # no guarantees that textureless stuff works
            sampler_count_before = -1

        # only support 1 subshape (& 1 sampler per) for now
        subshape = AGBSubshape(
            sampler_count=1 if tex_img else 0,
            unk_04=1, # ?
            tev_mode=0, # ?
            unk_0c=0, # ?
            sampler_indices=[sampler_count_before] + [-1] * 7,
            sampler_source_texture_coordinate_indices=[0 if tex_img else -1] + [-1] * 7, # ?
            polygon_base_index=polygon_count_before,
            polygon_count=len(shape_eval.data.polygons),

            vertex_position_indices_base_index=vtxpos_indices_count_before,
            vertex_normal_base_indices_base_index=vtxnrm_indices_count_before,
            vertex_color_base_indices_base_index=vtxclr_indices_count_before,
            vertex_texture_coordinate_indices_base_index=[vtxtexcoord_indices_count_before] + [0] * 7,
        )
        subshape_count_before = len(agb.subshapes)
        agb.subshapes.append(subshape)

        AGBshape = AGBShape(
            name=bone.name,

            vertex_position_data_base_index=vtx_count_before,
            vertex_position_data_count=len(shape_eval.data.vertices),
            vertex_normal_data_base_index=vtx_count_before,
            vertex_normal_data_count=len(shape_eval.data.vertices),
            vertex_color_data_base_index=vtxclr_count_before,
            vertex_color_data_count=len(shape_eval.data.vertices) if colors else 1,

            # only supporting one texcoord thingy right now
            vertex_texture_coordinate0_data_base_index=vtxtexcoord_count_before,
            vertex_texture_coordinate0_data_count=len(shape_eval.data.vertices) if uvs else 0,
            vertex_texture_coordinate1_data_base_index=0,
            vertex_texture_coordinate1_data_count=0,
            vertex_texture_coordinate2_data_base_index=0,
            vertex_texture_coordinate2_data_count=0,
            vertex_texture_coordinate3_data_base_index=0,
            vertex_texture_coordinate3_data_count=0,
            vertex_texture_coordinate4_data_base_index=0,
            vertex_texture_coordinate4_data_count=0,
            vertex_texture_coordinate5_data_base_index=0,
            vertex_texture_coordinate5_data_count=0,
            vertex_texture_coordinate6_data_base_index=0,
            vertex_texture_coordinate6_data_count=0,
            vertex_texture_coordinate7_data_base_index=0,
            vertex_texture_coordinate7_data_count=0,

            subshape_base_index=subshape_count_before,
            subshape_count=1,
            draw_mode=0, # ?
            cull_mode=0 if backface_culling else 3,
        )
        shape_count_before = len(agb.shapes)
        agb.shapes.append(AGBshape)

    anim_data = armature.animation_data
    if anim_data:
        for track in anim_data.nla_tracks:
            handle_nla_track(track, bone)
    
    obj_tsrf = get_tsrf(bone)

    if hasattr(shape_eval, "matrix_local"):
        mat = shape_eval.matrix_local
    else:
        mat = shape_eval.matrix
    
    translation = mat.to_translation()
    rotation = mat.to_euler()
    scale = mat.to_scale()
    group_transform = AGBGroupTransform(
        translation=agbvec(translation * SUPERRESOLUTION_FACTOR * obj_tsrf),
        scale=agbvec(scale * obj_tsrf),
        rotation_in_2deg=Vec3(
            degrees(rotation.x) / 2,
            degrees(rotation.y) / 2,
            degrees(rotation.z) / 2,
        ), # ???
        joint_post_rotation_in_deg=ZERO_VEC3, # ???

        # i have no clue
        transform_rotation_pivot=ZERO_VEC3,
        transform_scale_pivot=ZERO_VEC3,
        transform_rotation_offset=ZERO_VEC3,
        transform_scale_offset=ZERO_VEC3,
    )

    group = AGBGroup(
        name=bone.name,
        next_group_id=-1,
        child_group_id=-1,
        shape_id=shape_count_before if has_shape else -1,
        is_joint=0, # i guess?

        # placeholders (replaced later)
        visibility_group_id= 0,
        transform_base_index=0,
    )

    prev_child_group_ix = -1

    for child in reversed(bone.children):
        child_group_ix = object_to_group(child, ident+1)

        agb.groups[child_group_ix].next_group_id = prev_child_group_ix

        prev_child_group_ix = child_group_ix
    
    group.child_group_id = prev_child_group_ix

    group.transform_base_index = gti2fi(len(agb.group_transform_data))
    agb.group_transform_data.append(group_transform)

    # FIXME: There's no way to get the "non-animated" value of hide_render,
    # so this will always equal the value at frame 0 in the currently selected animation.
    # The recommended workaround is to set a keyframe at frame 0 that defines the
    # default visibility state in every animation that changes visibility.
    # I don't think this is fixable, but maybe there's a way to make it less of a pain?
    # (e.g. if the first visibility keyframe in all exported actions used by an object
    #  sets the object to hidden, then maybe the default state is visible)
    group.visibility_group_id = len(agb.visibility_groups)
    if has_shape:
        agb.visibility_groups.append(int(not shape.hide_render))
    else:
        agb.visibility_groups.append(int(True))

    group_count_before = len(agb.groups)
    agb.groups.append(group)
    obj_group_map[bone] = group_count_before

    if obj_tsrf != 1:
        scale_undo_grp = AGBGroup(
            name=f'scale_undo_{bone.name}',
            next_group_id=-1,
            child_group_id=group_count_before,
            shape_id=-1,
            # reusing the only children's visgroup is the best way to go, I think?
            visibility_group_id=group.visibility_group_id,
            transform_base_index=gti2fi(len(agb.group_transform_data)),
            is_joint=0,
        )

        scale = Vector((1.0, 1.0, 1.0)) / TRANSFORM_SUPERRESOLUTION_FACTOR
        agb.group_transform_data.append(
            AGBGroupTransform(
                translation=ZERO_VEC3,
                scale=agbvec(scale),
                rotation_in_2deg=ZERO_VEC3,
                joint_post_rotation_in_deg=ZERO_VEC3,
                transform_rotation_pivot=ZERO_VEC3,
                transform_scale_pivot=ZERO_VEC3,
                transform_rotation_offset=ZERO_VEC3,
                transform_scale_offset=ZERO_VEC3,
            )
        )

        group_count_before = len(agb.groups)
        agb.groups.append(scale_undo_grp)

    return group_count_before # cur group ix


def handle_nla_track(track, bone):
    if track.name[0] == '!' and len(track.strips) == 1:
        anim_name = track.name[1:]
        if anim_name not in nlatrack_map:
            nlatrack_map[anim_name] = {}

        assert(bone not in nlatrack_map[anim_name])

        action = track.strips[0].action
        nlatrack_map[anim_name][bone] = (track, action)

        for fcurve in action.fcurves.values():
            # we don't include rotation because it can't be upscaled
            if not fcurve.mute and fcurve.data_path in ['location', 'scale']:
                # the object has a scale transform animation,
                # so we need to apply the superres factor
                has_transform_anim_map[bone] = True
                break


# sampling everything for each frame
# i don't really like it but it's probably more reliable than the alternatives, if there's any
def bake_actions_to_anims():
    for anim_name, tracks in nlatrack_map.items():
        cur_anim = AGBAnimation(
            name=anim_name,
            data=AGBAnimationData(
                base_info=[AGBAnimationBaseInfo(
                  loop=1,
                  anim_start=math.inf,
                  anim_end=-math.inf,
                )],
                keyframes=[],
                vertex_position_deltas=[],
                vertex_normal_deltas=[],
                texture_coordinate_transform_deltas=[],
                visibility_group_deltas=[],
                group_transform_data_deltas=[],
                anim_data_type8_data=[bytes()]
            ),
        )

        base_info = cur_anim.data.base_info[0]

        # find anim frame range
        frame_start = 1
        for obj, (track, action) in tracks.items():
            frame_end = int(action.frame_range[1])

            if frame_start < base_info.anim_start: # currently useless, we assume all animations start at 1
                base_info.anim_start = frame_start
            if frame_end > base_info.anim_end:
                base_info.anim_end = frame_end

            # and while we're at it let's check for unsupported keyframe positions
            for fcurve in action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    frame = keyframe.co[0]

                    assert(frame == int(frame)) # is not float (we don't support sub-frames)

                    # see the comment block on visibility in object_to_group for details on this
                    assert(frame >= frame_start or fcurve.data_path == 'hide_render')

        # prebuild the keyframe list starting at 0, so that indexing by frame works
        # unused keyframes get thrown away at the end
        for frame in range(base_info.anim_end + 1):
            cur_anim.data.keyframes.append(
                AGBAnimationKeyframe(
                    # all values need to be filled here for the keyframe cleaning later (which I should really redo)
                    time=frame,
                    vertex_position_delta_base_index=0,
                    vertex_position_delta_count=0,
                    vertex_normal_delta_base_index=0,
                    vertex_normal_delta_count=0,
                    texture_coordinate_transform_delta_base_index=0,
                    texture_coordinate_transform_delta_count=0,
                    visibility_group_delta_base_index=0,
                    visibility_group_delta_count=0,
                    group_transform_data_delta_base_index=0,
                    group_transform_data_delta_count=0,
                )
            )

        # visibility animation
        frames_visibility = defaultdict(dict)
        for bone, (track, action) in tracks.items():
            group_ix = obj_group_map[bone]
            group = agb.groups[group_ix]

            for fcurve in action.fcurves:
                if fcurve.data_path != 'hide_render':
                    continue

                for keyframe in fcurve.keyframe_points:
                    frame, hidden = keyframe.co
                    if frame >= frame_start:
                        frames_visibility[int(frame)][group.visibility_group_id] = int(not hidden)

        for frame, vis_changes in frames_visibility.items():
            visgroup_delta_count_before = len(cur_anim.data.visibility_group_deltas)

            vis_changes = sorted(vis_changes.items())
            prev_visgroup_id = 0
            for visgroup_id, visible in vis_changes:
                prev_visible = None
                for old_frame in sorted((of for of in frames_visibility.keys() if of < frame), reverse=True):
                    old_vischanges = frames_visibility[old_frame]
                    if visgroup_id in old_vischanges:
                        prev_visible = old_vischanges[visgroup_id]
                        break

                if prev_visible is None:
                    prev_visible = agb.visibility_groups[visgroup_id]

                index_delta = visgroup_id - prev_visgroup_id

                while index_delta > 255:
                    index_delta -= 255
                    cur_anim.data.visibility_group_deltas.append(
                        AGBAnimationVisibilityGroupStatus(
                            index_delta=255,
                            visible=0,
                        )
                    )

                # we don't want to end up at 2 or -1
                # (note that noclip doesn't use addition but rather directly sets values,
                # so potential bugs stemming from this will only appear in the real game)
                if visible != prev_visible:
                    cur_anim.data.visibility_group_deltas.append(
                        AGBAnimationVisibilityGroupStatus(
                            index_delta=index_delta,
                            visible=1 if visible else -1,
                        )
                    )

                prev_visgroup_id = visgroup_id

            visgroup_delta_count_keyframe = len(cur_anim.data.visibility_group_deltas) - visgroup_delta_count_before
            if visgroup_delta_count_keyframe > 0:
                keyframe = cur_anim.data.keyframes[frame] # is a reference
                keyframe.visibility_group_delta_base_index = visgroup_delta_count_before
                keyframe.visibility_group_delta_count = visgroup_delta_count_keyframe
        cur_anim.data.visibility_group_delta_count = len(cur_anim.data.visibility_group_deltas)
        
        # for sampling, we need to use *only* the tracks related to the animation,
        # otherwise, other stuff would interfere and break it
        orig_muted_tracks = {}
        unlinked_actions = {}
        for obj in exported_collection.all_objects.values():
            if obj.animation_data is None:
                obj.animation_data_create()

            selected_track = None
            if any(bone in tracks for bone in obj.pose.bones):
            if obj.type == 'ARMATURE' and any(bone in tracks for bone in obj.pose.bones):
                selected_track = tracks[bone][0]

            for track in obj.animation_data.nla_tracks:
                orig_muted_tracks[track] = track.mute
                if selected_track and track == selected_track:
                    track.mute = False
                else:
                    track.mute = True

            action = obj.animation_data.action
            if action:
                unlinked_actions[obj] = (action, action.use_fake_user)
                if not action.use_fake_user:
                    # I don't believe this is actually required to prevent the action
                    # from disappearing, but I won't take any chances
                    action.use_fake_user = True
                obj.animation_data.action = None

        # group transform animation handling
        frames_grptrans = {}
        for bone, (track, action) in tracks.items():
            if not hasattr(bone, 'bone') or bone.bone is None:
                print(bone, " is not bone!")
                continue

            group_ix = obj_group_map[bone]
            group = agb.groups[group_ix]

            obj_tsrf = get_tsrf(bone)
            for frame in range(base_info.anim_start, int(action.frame_range[1]) + 1):
                if frame not in frames_grptrans:
                    frames_grptrans[frame] = {}

                scene.frame_set(frame)

                depsgraph = bpy.context.evaluated_depsgraph_get()

                bone_eval = armature.evaluated_get(depsgraph).pose.bones.get(bone.name)
                mat = bone_eval.matrix

                translation = mat.to_translation() * SUPERRESOLUTION_FACTOR * obj_tsrf
                rotation = Vector((degrees(r) / 2 for r in mat.to_euler()))
                scale = mat.to_scale() * obj_tsrf

                for vix, vec in enumerate([translation, scale, rotation]):
                    for cix in range(3):
                        transform_offset = group.transform_base_index + (vix * 3) + cix
                        # tangent_out is 90 to skip interpolation because it's been causing issues.
                        # we didn't use it anyway because we're sampling each frame, so there's no downside
                        frames_grptrans[frame][transform_offset] = (vec[cix], 0, 90)

        scene.frame_set(0)

        grptrans_deltas = {}
        grptrans_repeating = {}
        grptrans_repeating_insertions = []
        grptrans_incomplete_delta = {}
        for frame, transforms in frames_grptrans.items():
            transforms = sorted(transforms.items())
            if frame not in grptrans_deltas:
                grptrans_deltas[frame] = []

            for transform_offset, (value, tangent_in, tangent_out) in transforms:
                if frame == base_info.anim_start:
                    prev_value = flat_group_transforms[transform_offset]
                else:
                    prev_value = None
                    for old_frame in reversed(range(base_info.anim_start, frame)):
                        old_values = frames_grptrans[old_frame]
                        if transform_offset in old_values:
                            prev_value = old_values[transform_offset][0]
                            break

                value_delta = value - prev_value

                # prevent interpolation if a transform is supposed to stay still
                # (it can save a fair bit of space over always creating keyframes)
                # note: we're doing this *before* the rounding & 1/16 stuff because:
                # - if the non-0 "combo-breaker" delta is non-0 after rounding, then
                #   we get the same interpolation as if we did it after
                # - if it *is* 0 after rounding and there's no keyframes between the
                #   last "real" 0-delta and the next one with an actual value, we get
                #   better interpolation than if we did it after
                # - if there *is* a keyframe between these deltas, then we get the same
                #   interpolation as if we did it after (or better if not all keyframe
                #   "slots" between the deltas are filled)
                if value_delta == 0:
                    grptrans_repeating[transform_offset] = (frame, len(grptrans_deltas[frame]))
                    continue
                else:
                    ins = grptrans_repeating.get(transform_offset)
                    if ins is not None:
                        grptrans_repeating_insertions.append((transform_offset, *ins))
                        grptrans_repeating[transform_offset] = None

                # prevent drift (i.e. rounding error accumulation) as much as we can
                incomplete_delta = grptrans_incomplete_delta.get(transform_offset)
                if incomplete_delta is not None:
                    value_delta += incomplete_delta

                adjusted_delta = int(value_delta / (1/16))

                new_incomplete_delta = value_delta - (adjusted_delta * (1/16))
                grptrans_incomplete_delta[transform_offset] = new_incomplete_delta or None

                # account for deltas going past the s8 limits by segmenting them
                while adjusted_delta != 0:
                    cur_delta = max(-128, min(adjusted_delta, 127))
                    adjusted_delta -= cur_delta

                    grptrans_deltas[frame].append(
                        AGBAnimationGroupTransformDataDelta(
                            index_delta=transform_offset,
                            value_delta=cur_delta,
                            tangent_in_deg=tangent_in,
                            tangent_out_deg=tangent_out,
                        )
                    )

                prev_trans_offset = transform_offset

        # apply keep-still keyframes in reverse, so that earlier ones don't offset the position of the later ones
        for transform_offset, ins_frame, ins_ix in reversed(grptrans_repeating_insertions):
            grptrans_deltas[ins_frame].insert(
                ins_ix,
                AGBAnimationGroupTransformDataDelta(
                    index_delta=transform_offset,
                    value_delta=0,
                    tangent_in_deg=0,
                    tangent_out_deg=0,
                )
            )

        # make absolute indices relative
        # needs to be a separate pass to account for repeating_insertions
        for frame, deltas in grptrans_deltas.items():
            grptrans_delta_count_before = len(cur_anim.data.group_transform_data_deltas)

            prev_trans_offset = 0
            for delta in deltas:
                assert(delta.index_delta >= prev_trans_offset)

                index_delta = delta.index_delta - prev_trans_offset

                # prevent u8 overflow if necessary
                while index_delta > 255:
                    index_delta -= 255
                    cur_anim.data.group_transform_data_deltas.append(
                        AGBAnimationGroupTransformDataDelta(
                            index_delta=255,
                            value_delta=0,
                            tangent_in_deg=0,
                            tangent_out_deg=0,
                        )
                    )

                cur_anim.data.group_transform_data_deltas.append(
                    AGBAnimationGroupTransformDataDelta(
                        index_delta=index_delta,
                        value_delta=delta.value_delta,
                        tangent_in_deg=delta.tangent_in_deg,
                        tangent_out_deg=delta.tangent_out_deg,
                    )
                )

                prev_trans_offset = delta.index_delta

            grptrans_delta_count_keyframe = len(cur_anim.data.group_transform_data_deltas) - grptrans_delta_count_before
            if grptrans_delta_count_keyframe > 0:
                keyframe = cur_anim.data.keyframes[frame] # is a reference
                keyframe.group_transform_data_delta_base_index = grptrans_delta_count_before
                keyframe.group_transform_data_delta_count = grptrans_delta_count_keyframe
        cur_anim.data.group_transform_data_delta_count = len(cur_anim.data.group_transform_data_deltas)

        # find the anim's bbox
        # this needs to be run before disabling the transform fcurves, or the matrix_world will be incorrect
        anim_bbox_min = Vector()
        anim_bbox_max = Vector()
        for frame in range(base_info.anim_start, base_info.anim_end+1):
            scene.frame_set(frame)
            depsgraph = bpy.context.evaluated_depsgraph_get()

            for obj in exported_collection.all_objects.values():
                if obj.type != 'MESH':
                    continue

                obj_eval = obj.evaluated_get(depsgraph)
                for vertex in obj_eval.data.vertices.values():
                    vtxco_world = obj_eval.matrix_world @ vertex.co
                    for ix in range(3):
                        co = vtxco_world[ix]
                        anim_bbox_min[ix] = min(anim_bbox_min[ix], co)
                        anim_bbox_max[ix] = max(anim_bbox_max[ix], co)
        cur_anim.data.anim_bbox_min = agbvec(anim_bbox_min)
        cur_anim.data.anim_bbox_max = agbvec(anim_bbox_max)

        # reset before disabling fcurves
        scene.frame_set(0)

        # bug workaround: in some cases, vertex positions change ever so slightly when a parent object is moved.
        # this can sometimes be fixed by applying scale (Ctrl+A) but I don't even want to bother
        # (if you're a Blender developer and you're reading this, please look into it <3)
        disabled_fcurves = {}
        for (track, action) in tracks.values():
            disabled_fcurves[action] = []
            for ix, fcurve in action.fcurves.items():
                if not fcurve.mute and fcurve.data_path in {'location', 'rotation_euler', 'scale'}:
                    fcurve.mute = True
                    disabled_fcurves[action].append(ix)

        # sample all vertex attributes of all objects at each frame of the current animation.
        # we have to iterate over all objects because an armature's mesh could be anywhere in the hierarchy.
        # if the memory cost becomes too high, an alternative would be to find all the meshes that have an
        # armature modifier, but it might break some edge cases I don't know about, so for now, I'll leave it at that
        frames_vtx_attrs = dict(position=defaultdict(dict), normal=defaultdict(dict))
        for frame in range(base_info.anim_start, base_info.anim_end+1):
            scene.frame_set(frame)
            depsgraph = bpy.context.evaluated_depsgraph_get()

            for obj in exported_collection.all_objects.values():
                if obj.type != 'MESH':
                    continue

                obj_eval = obj.evaluated_get(depsgraph)

                # check that the vertex/face/etc count doesn't change at any point during the animation
                # (that'd happen with e.g. an animated mesh modifier)
                assert((len(obj_eval.data.vertices), len(obj_eval.data.polygons), len(obj_eval.data.loops)) == mesh_base_counts[obj])

                for vertex in obj_eval.data.vertices.values():
                    vtxid = vtxpath2id[repr(vertex)]
                    frames_vtx_attrs['position'][frame][vtxid] = vertex.co * SUPERRESOLUTION_FACTOR
                    if not DISABLE_ANIMATED_NORMALS:
                        frames_vtx_attrs['normal'][frame][vtxid] = copy(vertex.normal)

        # prevent mid-animation states from carrying over
        scene.frame_set(0)

        # restore pre-anim state
        for track, mute in orig_muted_tracks.items():
            track.mute = mute

        for obj, data in unlinked_actions.items():
            obj.animation_data.action = data[0]
            obj.animation_data.action.use_fake_user = data[1]

        for action, indices in disabled_fcurves.items():
            for ix in indices:
                action.fcurves[ix].mute = False


        # actual vtxattr deltas calculations
        for attr, frames_data in frames_vtx_attrs.items():
            vtx_attr_deltas = {}
            vtxid_repeating = {}
            vtxid_repeating_insertions = []
            vtxattr_incomplete_delta = {}

            for frame, attr_values in frames_data.items():
                attr_values = sorted(attr_values.items())
                if frame not in vtx_attr_deltas:
                    vtx_attr_deltas[frame] = []

                for vtxid, value in attr_values:
                    if frame == base_info.anim_start:
                        prev_value = vtxid_base_attrs[vtxid][attr]
                    else:
                        prev_value = None

                        # i forgot why this needs to be a full search rather than a single lookup
                        for old_frame in reversed(range(base_info.anim_start, frame)):
                            old_vtxattr = frames_data[old_frame]
                            if vtxid in old_vtxattr:
                                prev_value = old_vtxattr[vtxid]
                                break

                        assert(prev_value is not None) # new vertex? maybe modifiers? anyway that's bad

                    value_delta = value - prev_value

                    # prevent interpolation if an attribute doesn't change for a while
                    # (it can save a fair bit of space over always creating keyframes)
                    if value_delta == ZERO_VECTOR:
                        vtxid_repeating[vtxid] = (frame, len(vtx_attr_deltas[frame]))
                        continue
                    else:
                        ins = vtxid_repeating.get(vtxid)
                        if ins is not None:
                            vtxid_repeating_insertions.append((vtxid, *ins))
                            vtxid_repeating[vtxid] = None

                    # prevent drift (i.e. rounding error accumulation) as much as we can
                    incomplete_val_delta = vtxattr_incomplete_delta.get(vtxid)
                    if incomplete_val_delta is not None:
                        value_delta += incomplete_val_delta

                    coord_deltas = [0 if is_nan(x) else int(x / (1/16)) for x in value_delta] # how the fuck can a vertex be positioned at nan

                    list_val_delta = list(value_delta)
                    new_incomplete_delta = Vector(list_val_delta[ix] - (coord_deltas[ix] * (1/16)) for ix in range(3))
                    vtxattr_incomplete_delta[vtxid] = None if new_incomplete_delta == ZERO_VECTOR else new_incomplete_delta

                    # account for deltas going past the s8 limits by segmenting them
                    while coord_deltas != ZERO_LIST3:
                        cur_deltas = [0] * 3
                        for ix in range(3):
                            x = max(-128, min(coord_deltas[ix], 127))
                            cur_deltas[ix] += x
                            coord_deltas[ix] -= x

                        vtx_attr_deltas[frame].append(
                            AGBAnimationVectorDelta(
                                index_delta=vtxid,
                                coordinate_deltas=cur_deltas,
                            )
                        )

            # apply keep-still keyframes in reverse, so that earlier ones don't offset the position of the later ones
            for vtxid, ins_frame, ins_ix in reversed(vtxid_repeating_insertions):
                vtx_attr_deltas[ins_frame].insert(
                    ins_ix,
                    AGBAnimationVectorDelta(
                        index_delta=vtxid,
                        coordinate_deltas=ZERO_LIST3,
                    )
                )

            # make absolute indices relative
            # needs to be a separate pass to account for repeating_insertions
            anim_attr_deltas = getattr(cur_anim.data, f'vertex_{attr}_deltas')
            for frame, deltas in vtx_attr_deltas.items():
                vtx_attr_delta_count_before = len(anim_attr_deltas)

                prev_vtxid = 0
                for delta in deltas:
                    assert(delta.index_delta >= prev_vtxid)

                    vtxid_delta = delta.index_delta - prev_vtxid

                    # prevent u8 overflow if necessary
                    while vtxid_delta > 255:
                        vtxid_delta -= 255
                        anim_attr_deltas.append(
                            AGBAnimationVectorDelta(
                                index_delta=255,
                                coordinate_deltas=ZERO_LIST3,
                            )
                        )

                    anim_attr_deltas.append(
                        AGBAnimationVectorDelta(
                            index_delta=vtxid_delta,
                            coordinate_deltas=delta.coordinate_deltas,
                        )
                    )

                    prev_vtxid = delta.index_delta

                vtx_attr_delta_count_keyframe = len(anim_attr_deltas) - vtx_attr_delta_count_before
                if vtx_attr_delta_count_keyframe > 0:
                    keyframe = cur_anim.data.keyframes[frame] # is a reference
                    setattr(keyframe, f'vertex_{attr}_delta_base_index', vtx_attr_delta_count_before)
                    setattr(keyframe, f'vertex_{attr}_delta_count', vtx_attr_delta_count_keyframe)
            setattr(cur_anim.data, f'vertex_{attr}_delta_count', len(anim_attr_deltas))

        # cleanup empty or/and invalid keyframes
        for kix in reversed(range(len(cur_anim.data.keyframes))):
            # checks that at least one "count" field is >0
            keyframe_values = tuple(cur_anim.data.keyframes[kix].__dict__.values())
            has_data = any(keyframe_values[vix] > 0 for vix in (2, 4, 6, 8, 10))
            if not has_data or kix > base_info.anim_end or kix < base_info.anim_start:
                del cur_anim.data.keyframes[kix]

        cur_anim.data.keyframe_count = len(cur_anim.data.keyframes)
        agb.anims.append(cur_anim)


if __name__ == "__main__":
    print("\n\n --- STARTING EXPORT SCRIPT --- \n\n")

    agb = AGBFile(
        header=AGBHeader(
            fixed_size_data_end=0, # this is automatically adjusted by the serializer
            model_name=MODEL_NAME,
            texture_name=MODEL_NAME,
            build_time=datetime.utcnow().strftime('%a %b %d %H:%M:%S %Y'),
            flags=1, # ?
            radius=5, # ?
            height=5, # ?
            bbox_min=Vector(),
            bbox_max=Vector(),
        ),

        shapes=[],
        polygons=[],

        vertex_positions=[],
        vertex_position_indices=[],
        vertex_normals=[],
        vertex_normal_indices=[],
        vertex_colors=[],
        vertex_color_indices=[],

        vertex_texture_coordinate_indices=[],

        vertex_texture_coordinates=[],
        texture_coordinate_transforms=[],

        samplers=[],
        textures=[],
        subshapes=[],

        visibility_groups=[],
        group_transform_data=[],
        groups=[],

        anims=[],
    )

    # globals, because it works
    nlatrack_map = {}
    obj_group_map = {}
    vtxpath2id = {}
    vtxid_base_attrs = {}
    mesh_base_counts = {}
    has_transform_anim_map = {}
    tpl_image_map = {}

    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')

    # requires that for anims, at frame 0, the model is no different than when the action is not active
    # if that condition is not fullfilled, well, it's all downhill from here :)
    # edit: might be fine actually. i'm not sure. i'm tired. let's hope for the best.
    scene = bpy.context.scene
    orig_frame = scene.frame_current
    scene.frame_set(0)

    collection = bpy.data.collections[MODEL_NAME]

    # get first armature
    armature = next((obj for obj in collection.all_objects if obj.type == 'ARMATURE'), None)
    assert (armature is not None)
    assert (armature.type == 'ARMATURE')

    exported_collection = collection

    orig_pose_positions = {}
    orig_pose_positions[armature] = armature.data.pose_position
    armature.data.pose_position = 'REST'

    # make AGB groups and everything they're related to (except for animations) from blender objects
    root_objects = [o for o in armature.pose.bones if not o.parent]    

    prev_root_obj_group_ix = -1
    for object in reversed(root_objects):
        root_obj_group_ix = object_to_group(object)

        agb.groups[root_obj_group_ix].next_group_id = prev_root_obj_group_ix

        prev_root_obj_group_ix = root_obj_group_ix

    agb.header.bbox_min = agbvec(agb.header.bbox_min)
    agb.header.bbox_max = agbvec(agb.header.bbox_max)

    for obj, pose_position in orig_pose_positions.items():
        obj.data.pose_position = pose_position

    flat_group_transforms = []
    for transform in agb.group_transform_data:
        for vec in transform.__dict__.values():
            flat_group_transforms.extend(vec.__dict__.values())

    # superres root
    if SUPERRESOLUTION_FACTOR != 1:
        agb.groups.append(
            AGBGroup(
                name='superres_scale_undo',
                next_group_id=-1,
                child_group_id=root_obj_group_ix,
                shape_id=-1,
                visibility_group_id=len(agb.visibility_groups),
                transform_base_index=len(flat_group_transforms),
                is_joint=0,
            )
        )

        agb.visibility_groups.append(1)

        scale = Vector((1.0, 1.0, 1.0)) / SUPERRESOLUTION_FACTOR
        agb.group_transform_data.append(
            AGBGroupTransform(
                translation=ZERO_VEC3,
                scale=agbvec(scale),
                rotation_in_2deg=ZERO_VEC3,
                joint_post_rotation_in_deg=ZERO_VEC3,
                transform_rotation_pivot=ZERO_VEC3,
                transform_scale_pivot=ZERO_VEC3,
                transform_rotation_offset=ZERO_VEC3,
                transform_scale_offset=ZERO_VEC3,
            )
        )

    bake_actions_to_anims()

    # restore the frame we were at originally
    scene.frame_set(orig_frame)

    armature.data.pose_position = 'POSE'

    with open(OUTPUT_PATH, 'wb') as f:
        agb_write(agb, f)
    
    print("AGB export complete!")