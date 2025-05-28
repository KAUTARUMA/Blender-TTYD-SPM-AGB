# Huge thanks to PistonMiner, as this is essentially adapted 1:1 from their documentation:
# https://github.com/PistonMiner/ttyd-tools/blob/master/ttyd-tools/docs/MarioSt_AnimGroupBase.bt
import struct
from struct import Struct
from typing import Optional, Union
from dataclasses import dataclass, field

@dataclass
class Vec2:
	x: float
	y: float

@dataclass
class Vec3:
	x: float
	y: float
	z: float

@dataclass
class Color4:
	r: int
	g: int
	b: int
	a: int

@dataclass
class AGBShape:
	name: str
	vertex_position_data_base_index: int
	vertex_position_data_count: int
	vertex_normal_data_base_index: int
	vertex_normal_data_count: int
	vertex_color_data_base_index: int
	vertex_color_data_count: int
	vertex_texture_coordinate0_data_base_index: int
	vertex_texture_coordinate0_data_count: int
	vertex_texture_coordinate1_data_base_index: int
	vertex_texture_coordinate1_data_count: int
	vertex_texture_coordinate2_data_base_index: int
	vertex_texture_coordinate2_data_count: int
	vertex_texture_coordinate3_data_base_index: int
	vertex_texture_coordinate3_data_count: int
	vertex_texture_coordinate4_data_base_index: int
	vertex_texture_coordinate4_data_count: int
	vertex_texture_coordinate5_data_base_index: int
	vertex_texture_coordinate5_data_count: int
	vertex_texture_coordinate6_data_base_index: int
	vertex_texture_coordinate6_data_count: int
	vertex_texture_coordinate7_data_base_index: int
	vertex_texture_coordinate7_data_count: int
	subshape_base_index: int
	subshape_count: int
	draw_mode: int
	cull_mode: int

@dataclass
class AGBPolygon:
	vertex_base_index: int
	vertex_count: int

@dataclass
class AGBTextureCoordinateTransform:
	texture_frame_offset: int
	translation_x: float
	translation_y: float
	scale_x: float
	scale_y: float
	rotation: float

@dataclass
class AGBSampler:
	texture_base_id: int
	wrap_flags: int

@dataclass
class AGBTexture:
	unk_0: int
	tpl_index: int
	wbUnused: int
	unk_c: str
	unk_38: tuple[int, int]

@dataclass
class AGBSubshape:
	sampler_count: int
	unk_04: int
	tev_mode: int
	unk_0c: int
	sampler_indices: list[int]
	sampler_source_texture_coordinate_indices: list[int]
	polygon_base_index: int
	polygon_count: int

	vertex_position_indices_base_index: int
	vertex_normal_base_indices_base_index: int
	vertex_color_base_indices_base_index: int
	vertex_texture_coordinate_indices_base_index: list[int]

@dataclass
class AGBGroupTransform:
	translation: Vec3
	scale: Vec3
	rotation_in_2deg: Vec3
	joint_post_rotation_in_deg: Vec3
	transform_rotation_pivot: Vec3
	transform_scale_pivot: Vec3
	transform_rotation_offset: Vec3
	transform_scale_offset: Vec3

@dataclass
class AGBGroup:
	name: str
	next_group_id: int
	child_group_id: int
	shape_id: int
	visibility_group_id: int
	transform_base_index: int
	is_joint: int

	@property
	def corrected_transform_index(self):
		return fi2gti(self.transform_base_index)

@dataclass
class AGBAnimationBaseInfo:
	loop: int
	anim_start: float
	anim_end: float

@dataclass
class AGBAnimationKeyframe:
	time: float
	vertex_position_delta_base_index: int
	vertex_position_delta_count: int
	vertex_normal_delta_base_index: int
	vertex_normal_delta_count: int
	texture_coordinate_transform_delta_base_index: int
	texture_coordinate_transform_delta_count: int
	visibility_group_delta_base_index: int
	visibility_group_delta_count: int
	group_transform_data_delta_base_index: int
	group_transform_data_delta_count: int

@dataclass
class AGBAnimationVectorDelta:
	index_delta: int
	coordinate_deltas: tuple[int, int, int]

@dataclass
class AGBAnimationTextureCoordinateTransformDelta:
	index_delta: int
	frame_ext_delta: int
	translate_x_delta: float
	translate_y_delta: float

@dataclass
class AGBAnimationVisibilityGroupStatus:
	index_delta: int
	visible: int

@dataclass
class AGBAnimationGroupTransformDataDelta:
	index_delta: int
	value_delta: int
	tangent_in_deg: int
	tangent_out_deg: int

@dataclass
class AGBAnimationData:
	data_size: int = 0
	base_info_count: int = 0
	keyframe_count: int = 0
	vertex_position_delta_count: int = 0
	vertex_normal_delta_count: int = 0
	texture_coordinate_transform_delta_count: int = 0
	visibility_group_delta_count: int = 0
	group_transform_data_delta_count: int = 0
	anim_data_type8_count: int = 0
	
	base_info_offset: int = 0
	keyframes_offset: int = 0
	vertex_position_deltas_offset: int = 0
	vertex_normal_deltas_offset: int = 0
	texture_coordinate_transform_deltas_offset: int = 0
	visibility_group_deltas_offset: int = 0
	group_transform_data_deltas_offset: int = 0
	anim_data_type8_data_offset: int = 0
	
	anim_bbox_min: 'Vec3' = field(default_factory=lambda: Vec3(0, 0, 0))
	anim_bbox_max: 'Vec3' = field(default_factory=lambda: Vec3(0, 0, 0))
	
	base_info: Optional[list[AGBAnimationBaseInfo]] = None
	keyframes: Optional[list[AGBAnimationKeyframe]] = None
	vertex_position_deltas: Optional[list[AGBAnimationVectorDelta]] = None
	vertex_normal_deltas: Optional[list[AGBAnimationVectorDelta]] = None
	texture_coordinate_transform_deltas: Optional[list[AGBAnimationTextureCoordinateTransformDelta]] = None
	visibility_group_deltas: Optional[list[AGBAnimationVisibilityGroupStatus]] = None
	group_transform_data_deltas: Optional[list[AGBAnimationGroupTransformDataDelta]] = None
	anim_data_type8_data: Optional[list[bytes]] = None

@dataclass
class AGBAnimation:
	name: str
	data_offset: int = 0
	data: Optional[AGBAnimationData] = None

@dataclass
class AGBHeader:
	fixed_size_data_end: int
	model_name: str
	texture_name: str
	build_time: str
	flags: int
	radius: int
	height: int
	bbox_min: Vec3
	bbox_max: Vec3

	shape_count: int = 0
	polygon_count: int = 0
	vertex_position_count: int = 0
	vertex_position_index_count: int = 0
	vertex_normal_count: int = 0
	vertex_normal_index_count: int = 0
	vertex_color_count: int = 0
	vertex_color_index_count: int = 0
	vertex_texture_coordinate0_index_count: int = 0
	vertex_texture_coordinate1_index_count: int = 0
	vertex_texture_coordinate2_index_count: int = 0
	vertex_texture_coordinate3_index_count: int = 0
	vertex_texture_coordinate4_index_count: int = 0
	vertex_texture_coordinate5_index_count: int = 0
	vertex_texture_coordinate6_index_count: int = 0
	vertex_texture_coordinate7_index_count: int = 0
	vertex_texture_coordinate_count: int = 0
	texture_coordinate_transform_count: int = 0
	sampler_count: int = 0
	texture_count: int = 0
	subshape_count: int = 0
	visibility_group_count: int = 0
	group_transform_data_count: int = 0
	group_count: int = 0
	anim_count: int = 0
	
	shapes_offset: int = 0
	polygons_offset: int = 0
	vertex_positions_offset: int = 0
	vertex_position_indices_offset: int = 0
	vertex_normals_offset: int = 0
	vertex_normal_indices_offset: int = 0
	vertex_colors_offset: int = 0
	vertex_color_indices_offset: int = 0
	vertex_texture_coordinate0_indices_offset: int = 0
	vertex_texture_coordinate1_indices_offset: int = 0
	vertex_texture_coordinate2_indices_offset: int = 0
	vertex_texture_coordinate3_indices_offset: int = 0
	vertex_texture_coordinate4_indices_offset: int = 0
	vertex_texture_coordinate5_indices_offset: int = 0
	vertex_texture_coordinate6_indices_offset: int = 0
	vertex_texture_coordinate7_indices_offset: int = 0
	vertex_texture_coordinates_offset: int = 0
	texture_coordinate_transforms_offset: int = 0
	samplers_offset: int = 0
	textures_offset: int = 0
	subshapes_offset: int = 0
	visibility_groups_offset: int = 0
	group_transform_data_offset: int = 0
	groups_offset: int = 0
	anims_offset: int = 0

@dataclass
class AGBFile:
	header: AGBHeader

	shapes: list[AGBShape]
	polygons: list[AGBPolygon]

	vertex_positions: list[Vec3]
	vertex_position_indices: list[int]
	vertex_normals: list[Vec3]
	vertex_normal_indices: list[int]
	vertex_colors: list[Color4]
	vertex_color_indices: list[int]

	vertex_texture_coordinate_indices: list[int]

	vertex_texture_coordinates: list[Vec2]
	texture_coordinate_transforms: list[AGBTextureCoordinateTransform]

	samplers: list[AGBSampler]
	textures: list[AGBTexture]
	subshapes: list[AGBSubshape]

	visibility_groups: list[int]
	group_transform_data: list[AGBGroupTransform]
	groups: list[AGBGroup]

	anims: list[AGBAnimation]


SIZEOF = {
	Vec2: 0x8,
	Vec3: 0xC,
	Color4: 0x4,
	AGBHeader: 0x1B0,
	AGBShape: 0xA8,
	AGBPolygon: 0x8,
	AGBTextureCoordinateTransform: 0x18,
	AGBSampler: 0x8,
	AGBTexture: 0x40,
	AGBSubshape: 0x6C,
	AGBGroupTransform: 0x60,
	AGBGroup: 0x58,
	AGBAnimation: 0x40,
	AGBAnimationData: 0x5C,
	AGBAnimationBaseInfo: 0xC,
	AGBAnimationKeyframe: 0x2C,
	AGBAnimationVectorDelta: 0x4,
	AGBAnimationTextureCoordinateTransformDelta: 0xC,
	AGBAnimationVisibilityGroupStatus: 0x2,
	AGBAnimationGroupTransformDataDelta: 0x4,
}

def gti2fi(i):
	return i * SIZEOF[AGBGroupTransform] // 4
def fi2gti(i):
	return i * 4 // SIZEOF[AGBGroupTransform]

def roundup(x, n):
	return x if x % n == 0 else ((x + n) - (x % n))


def agb_read(io):
	def u(fmt):
		fmt = Struct(f'>{fmt}')
		return fmt.unpack_from(io.read(fmt.size))
	
	def ul(fmt):
		return list(u(fmt))
	
	def us(length):
		return u(f'{length}s')[0].decode('shift_jis').rstrip('\0')

	header = AGBHeader(
		*u('I'),
		us(64),
		us(64),
		us(64),
		*u('3I'),
		Vec3(*u('3f')),
		Vec3(*u('3f')),
		*u('50I'),
	)

	for n in range(1, 8):
		assert(getattr(header, f'vertex_texture_coordinate{n}_index_count') == 0)

	io.seek(header.shapes_offset)
	shapes = [AGBShape(us(64), *u('26I')) for _ in range(header.shape_count)]

	io.seek(header.polygons_offset)
	polygons = [AGBPolygon(*u('2I')) for _ in range(header.polygon_count)]

	io.seek(header.vertex_positions_offset)
	vertex_positions = [Vec3(*u('3f')) for _ in range(header.vertex_position_count)]

	io.seek(header.vertex_position_indices_offset)
	vertex_position_indices = ul(f'{header.vertex_position_index_count}I')

	io.seek(header.vertex_normals_offset)
	vertex_normals = [Vec3(*u('3f')) for _ in range(header.vertex_normal_count)]
	io.seek(header.vertex_normal_indices_offset)
	vertex_normal_indices = ul(f'{header.vertex_normal_index_count}I')

	io.seek(header.vertex_colors_offset)
	vertex_colors = [Color4(*u('4B')) for _ in range(header.vertex_color_count)]
	io.seek(header.vertex_color_indices_offset)
	vertex_color_indices = ul(f'{header.vertex_color_index_count}I')

	# TODO: use all texcoords
	io.seek(header.vertex_texture_coordinate0_indices_offset)
	vertex_texture_coordinate_indices = list(u(f'{header.vertex_texture_coordinate0_index_count}I'))

	io.seek(header.vertex_texture_coordinates_offset)
	vertex_texture_coordinates = [Vec2(*u('2f')) for _ in range(header.vertex_texture_coordinate_count)]

	io.seek(header.texture_coordinate_transforms_offset)
	texture_coordinate_transforms = [AGBTextureCoordinateTransform(*u('B 3x 5f')) for _ in range(header.texture_coordinate_transform_count)]

	io.seek(header.samplers_offset)
	samplers = [AGBSampler(*u('2I')) for _ in range(header.sampler_count)]

	io.seek(header.textures_offset)
	textures = [AGBTexture(*u('3I'), us(44), u('2I')) for _ in range(header.texture_count)]
	for texture in textures:
		assert(texture.unk_0 == 0)
		assert(texture.wbUnused == 0)
		assert(texture.unk_38 == (0, 0))

	io.seek(header.subshapes_offset)
	subshapes = [AGBSubshape(*u('4I'), ul('8i'), ul('8b'), *u('5I'), ul('8I')) for _ in range(header.subshape_count)]
	for subshape in subshapes:
		assert(subshape.unk_0c == 0)
	
	io.seek(header.visibility_groups_offset)
	visibility_groups = ul(f'{header.visibility_group_count}B')

	io.seek(header.group_transform_data_offset)
	group_transform_data = [AGBGroupTransform(*[Vec3(*u('3f')) for _ in range(8)]) for _ in range(fi2gti(header.group_transform_data_count))]

	io.seek(header.groups_offset)
	groups = [AGBGroup(us(64), *u('3i 3I')) for _ in range(header.group_count)]

	io.seek(header.anims_offset)
	anims = [AGBAnimation(us(60), *u('I')) for _ in range(header.anim_count)]
	for anim in anims:
		if anim.data_offset == 0:
			continue
		
		io.seek(anim.data_offset)
		anim.data = AGBAnimationData(*u('17I'), *[Vec3(*u('3f')) for _ in range(2)])
		assert(anim.data.base_info_offset == SIZEOF[AGBAnimationData])

		if anim.data.base_info_offset > 0:
			io.seek(anim.data_offset + anim.data.base_info_offset)
			anim.data.base_info = [AGBAnimationBaseInfo(*u('I 2f')) for _ in range(anim.data.base_info_count)]

		if anim.data.keyframes_offset > 0:
			io.seek(anim.data_offset + anim.data.keyframes_offset)
			anim.data.keyframes = [AGBAnimationKeyframe(*u('f 10I')) for _ in range(anim.data.keyframe_count)]
		
		if anim.data.vertex_position_deltas_offset > 0:
			io.seek(anim.data_offset + anim.data.vertex_position_deltas_offset)
			anim.data.vertex_position_deltas = [
				AGBAnimationVectorDelta(*u('B'), u('3b'))
				for _ in range(anim.data.vertex_position_delta_count)
			]
		
		if anim.data.vertex_normal_deltas_offset > 0:
			io.seek(anim.data_offset + anim.data.vertex_normal_deltas_offset)
			anim.data.vertex_normal_deltas = [
				AGBAnimationVectorDelta(*u('B'), u('3b'))
				for _ in range(anim.data.vertex_normal_delta_count)
			]
		
		if anim.data.texture_coordinate_transform_deltas_offset > 0:
			io.seek(anim.data_offset + anim.data.texture_coordinate_transform_deltas_offset)
			anim.data.texture_coordinate_transform_deltas = [
				AGBAnimationTextureCoordinateTransformDelta(*u('B b 2x 2f'))
				for _ in range(anim.data.texture_coordinate_transform_delta_count)
			]

		if anim.data.visibility_group_deltas_offset > 0:
			io.seek(anim.data_offset + anim.data.visibility_group_deltas_offset)
			anim.data.visibility_group_deltas = [
				AGBAnimationVisibilityGroupStatus(*u('B b'))
				for _ in range(anim.data.visibility_group_delta_count)
			]

		if anim.data.group_transform_data_deltas_offset > 0:
			io.seek(anim.data_offset + anim.data.group_transform_data_deltas_offset)
			anim.data.group_transform_data_deltas = [
				AGBAnimationGroupTransformDataDelta(*u('B 3b'))
				for _ in range(anim.data.group_transform_data_delta_count)
			]

		if anim.data.anim_data_type8_data_offset > 0:
			io.seek(anim.data_offset + anim.data.anim_data_type8_data_offset)
			anim.data.anim_data_type8_data = [
				u('8s')[0]
				for _ in range(anim.data.anim_data_type8_count)
			]


	return AGBFile(
		header,
		shapes,
		polygons,

		vertex_positions,
		vertex_position_indices,
		vertex_normals,
		vertex_normal_indices,
		vertex_colors,
		vertex_color_indices,
		vertex_texture_coordinate_indices,
		vertex_texture_coordinates,

		texture_coordinate_transforms,

		samplers,
		textures,
		subshapes,

		visibility_groups,
		group_transform_data,
		groups,

		anims,
	)

def agb_write(agb: AGBFile, io):
	def p(fmt, *args):
		packed = struct.pack(f'>{fmt}', *args)
		io.write(packed)
	def pd(fmt, obj):
		values = []
		if hasattr(obj, '__dict__'):
			for key, val in obj.__dict__.items():
				# is Optional
				annotation = obj.__annotations__[key]
				if (hasattr(annotation, '__origin__') and
					annotation.__origin__ == Union and
					annotation.__args__[1] == type(None)):
					continue
				
				if isinstance(val, str):
					val = val.encode('shift_jis')
				
				if isinstance(val, (list, tuple)):
					values.extend(val)
				elif isinstance(val, (Vec2, Vec3, Color4)):
					values.extend(val.__dict__.values())
				else:
					values.append(val)
		else:
			values = [obj]
		
		p(fmt, *values)
	def pds(fmt, objs):
		for obj in objs:
			pd(fmt, obj)
	def align(n):
		byte_count = n - (io.tell() % n)
		if byte_count != n:
			return io.write(b'\0' * byte_count)
		return 0
	
	io.seek(SIZEOF[AGBHeader])
	agb.header.shape_count = len(agb.shapes)
	agb.header.shapes_offset = io.tell()
	pds('64s 26I', agb.shapes)
	
	agb.header.polygon_count = len(agb.polygons)
	agb.header.polygons_offset = io.tell()
	pds('2I', agb.polygons)

	agb.header.vertex_position_count = len(agb.vertex_positions)
	agb.header.vertex_positions_offset = io.tell()
	pds('3f', agb.vertex_positions)
	agb.header.vertex_position_index_count = len(agb.vertex_position_indices)
	agb.header.vertex_position_indices_offset = io.tell()
	pds('I', agb.vertex_position_indices)

	agb.header.vertex_normal_count = len(agb.vertex_normals)
	agb.header.vertex_normals_offset = io.tell()
	pds('3f', agb.vertex_normals)
	agb.header.vertex_normal_index_count = len(agb.vertex_normal_indices)
	agb.header.vertex_normal_indices_offset = io.tell()
	pds('I', agb.vertex_normal_indices)

	agb.header.vertex_color_count = len(agb.vertex_colors)
	agb.header.vertex_colors_offset = io.tell()
	pds('4B', agb.vertex_colors)
	agb.header.vertex_color_index_count = len(agb.vertex_color_indices)
	agb.header.vertex_color_indices_offset = io.tell()
	pds('I', agb.vertex_color_indices)

	agb.header.vertex_texture_coordinate0_index_count = len(agb.vertex_texture_coordinate_indices)
	agb.header.vertex_texture_coordinate0_indices_offset = io.tell()
	pds('I', agb.vertex_texture_coordinate_indices)

	agb.header.vertex_texture_coordinate_count = len(agb.vertex_texture_coordinates)
	agb.header.vertex_texture_coordinates_offset = io.tell()
	pds('2f', agb.vertex_texture_coordinates)

	agb.header.texture_coordinate_transform_count = len(agb.texture_coordinate_transforms)
	agb.header.texture_coordinate_transforms_offset = io.tell()
	pds('B 3x 5f', agb.texture_coordinate_transforms)

	agb.header.sampler_count = len(agb.samplers)
	agb.header.samplers_offset = io.tell()
	pds('2I', agb.samplers)

	agb.header.texture_count = len(agb.textures)
	agb.header.textures_offset = io.tell()
	pds('3I 44s 2I', agb.textures)

	agb.header.subshape_count = len(agb.subshapes)
	agb.header.subshapes_offset = io.tell()
	pds('4I 8i 8b 13I', agb.subshapes)

	agb.header.visibility_group_count = len(agb.visibility_groups)
	agb.header.visibility_groups_offset = io.tell()
	pds('B', agb.visibility_groups)
	align(4)

	agb.header.group_transform_data_count = gti2fi(len(agb.group_transform_data))
	agb.header.group_transform_data_offset = io.tell()
	pds('24f', agb.group_transform_data)

	agb.header.group_count = len(agb.groups)
	agb.header.groups_offset = io.tell()
	pds('64s 3i 3I', agb.groups)

	agb.header.anim_count = len(agb.anims)
	agb.header.anims_offset = io.tell()
	agb.header.fixed_size_data_end = agb.header.anims_offset + (agb.header.anim_count * SIZEOF[AGBAnimation])

	cur_anim_data_end = agb.header.fixed_size_data_end
	for anim in agb.anims:
		tell = io.tell()
		anim.data_offset = 0
		if anim.data:
			anim.data_offset = cur_anim_data_end
			io.seek(anim.data_offset + SIZEOF[AGBAnimationData])

			anim.data.base_info_count = len(anim.data.base_info)
			anim.data.base_info_offset = io.tell() - anim.data_offset
			pds('I 2f', anim.data.base_info)

			anim.data.keyframe_count = len(anim.data.keyframes)
			anim.data.keyframes_offset = io.tell() - anim.data_offset
			pds('f 10I', anim.data.keyframes)

			anim.data.vertex_position_delta_count = len(anim.data.vertex_position_deltas)
			anim.data.vertex_position_deltas_offset = io.tell() - anim.data_offset
			pds('B 3b', anim.data.vertex_position_deltas)

			anim.data.vertex_normal_delta_count = len(anim.data.vertex_normal_deltas)
			anim.data.vertex_normal_deltas_offset = io.tell() - anim.data_offset
			pds('B 3b', anim.data.vertex_normal_deltas)

			anim.data.texture_coordinate_transform_delta_count = len(anim.data.texture_coordinate_transform_deltas)
			anim.data.texture_coordinate_transform_deltas_offset = io.tell() - anim.data_offset
			pds('B b 2x 2f', anim.data.texture_coordinate_transform_deltas)

			anim.data.visibility_group_delta_count = len(anim.data.visibility_group_deltas)
			anim.data.visibility_group_deltas_offset = io.tell() - anim.data_offset
			pds('B b', anim.data.visibility_group_deltas)
			align(4)

			anim.data.group_transform_data_delta_count = len(anim.data.group_transform_data_deltas)
			anim.data.group_transform_data_deltas_offset = io.tell() - anim.data_offset
			pds('B 3b', anim.data.group_transform_data_deltas)

			anim.data.anim_data_type8_count = len(anim.data.anim_data_type8_data)
			anim.data.anim_data_type8_data_offset = io.tell() - anim.data_offset
			pds('8s', anim.data.anim_data_type8_data)

			cur_anim_data_end = io.tell()
			anim.data.data_size = cur_anim_data_end - anim.data_offset

			io.seek(anim.data_offset)
			pd('17I 6f', anim.data)
		io.seek(tell)
		pd('60s I', anim)
	
	io.seek(0)
	pd('I 64s 64s 64s 3I 6f 50I', agb.header)