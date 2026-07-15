#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "device.hpp"
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

// Typecode for Python's built-in `array` module matching `type`, used to
// accept a plain list/tuple for Buffer::write() on a VECTOR/MATRIX field
// (see iterable_to_buffer below). No dependency on numpy: `array.array`
// already implements the buffer protocol Buffer::write() expects.
char scalar_type_to_array_typecode(ScalarType type) {
	switch (type) {
		case ScalarType::FLOAT32: return 'f';
		case ScalarType::FLOAT64: return 'd';
		case ScalarType::INT8: return 'b';
		case ScalarType::UINT8: return 'B';
		case ScalarType::INT16: return 'h';
		case ScalarType::UINT16: return 'H';
		case ScalarType::INT32: return 'i';
		case ScalarType::UINT32: return 'I';
		case ScalarType::INT64: return 'q';
		case ScalarType::UINT64: return 'Q';
		default:
			throw std::runtime_error(
				"write(): a plain list/tuple isn't supported for this scalar type; pass a numpy array or torch tensor instead");
	}
}

// Recursively flattens a nested Python list/tuple (matching a vector's
// flat shape, or a matrix's row-major nested shape) into `out`, preserving
// each leaf's original Python number so array.array() below converts it
// itself (no precision loss going through an intermediate C++ numeric type).
void flatten_iterable_into(const py::handle& value, py::list& out) {
	if (py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value)) {
		for (auto item : value) {
			flatten_iterable_into(item, out);
		}
	} else {
		out.append(value);
	}
}

// Converts a plain Python list/tuple `value` (for a VECTOR or MATRIX field
// with scalar component type `component_type`) into a Python array.array
// of the matching typecode, tightly packed in the same row-major order a
// numpy array of the same nested shape would export via the buffer
// protocol -- so it can be hand off directly to the existing, unmodified
// Buffer::write() without any change to its C++ implementation.
py::object iterable_to_buffer(ScalarType component_type, const py::object& value) {
	py::list flat;
	flatten_iterable_into(value, flat);
	const char typecode = scalar_type_to_array_typecode(component_type);
	py::object array_module = py::module_::import("array");
	return array_module.attr("array")(std::string(1, typecode), flat);
}

} // namespace

PYBIND11_MODULE(vk, m) {
	m.doc() = "Minimal Vulkan bindings";

	py::enum_<MemoryLocation>(m, "MemoryLocation")
		.value("HOST", MemoryLocation::HOST)
		.value("DEVICE", MemoryLocation::DEVICE)
		.export_values();

	// C++ enumerator names avoid the literal IN/OUT (Windows headers
	// #define these), but the Python-facing names are exactly IN/OUT/INOUT.
	py::enum_<WrapMode>(m, "WrapMode")
		.value("IN", WrapMode::CopyIn)
		.value("OUT", WrapMode::CopyOut)
		.value("INOUT", WrapMode::CopyInOut)
		.export_values();

	py::enum_<ScalarType>(m, "ScalarType")
	.value("UNDEFINED", ScalarType::UNDEFINED)
	.value("BOOL", ScalarType::BOOL)
	.value("FLOAT16", ScalarType::FLOAT16)
	.value("FLOAT32", ScalarType::FLOAT32)
	.value("FLOAT64", ScalarType::FLOAT64)
	.value("INT8", ScalarType::INT8)
	.value("INT16", ScalarType::INT16)
	.value("INT32", ScalarType::INT32)
	.value("INT64", ScalarType::INT64)
	.value("UINT8", ScalarType::UINT8)
	.value("UINT16", ScalarType::UINT16)
	.value("UINT32", ScalarType::UINT32)
	.value("UINT64", ScalarType::UINT64)
	.export_values();

	py::enum_<EngineType>(m, "EngineType")
		.value("NONE", EngineType::NONE)
		.value("GRAPHICS", EngineType::GRAPHICS)
		.value("COMPUTE", EngineType::COMPUTE)
		.value("TRANSFER", EngineType::TRANSFER)
		.export_values();

	py::enum_<TypeKind>(m, "TypeKind")
		.value("SCALAR", TypeKind::SCALAR)
		.value("VECTOR", TypeKind::VECTOR)
		.value("MATRIX", TypeKind::MATRIX)
		.value("ARRAY", TypeKind::ARRAY)
		.value("STRUCT", TypeKind::STRUCT)
		.export_values();

	py::enum_<LayoutRule>(m, "LayoutRule")
		.value("Std140", LayoutRule::Std140)
		.value("Std430", LayoutRule::Std430)
		.value("Scalar", LayoutRule::Scalar)
		.export_values();

	// Not .export_values() here: PipelineType/ShaderStageType both declare a
	// COMPUTE member, which would otherwise collide with each other (and with
	// EngineType.COMPUTE, above) in the module's flat namespace. All existing
	// call sites already use qualified access (e.g. vk.EngineType.COMPUTE).
	py::enum_<PipelineType>(m, "PipelineType")
		.value("COMPUTE", PipelineType::COMPUTE)
		.value("RASTERIZATION", PipelineType::RASTERIZATION)
		.value("RAYTRACING", PipelineType::RAYTRACING);

	py::enum_<ShaderStageType>(m, "ShaderStageType")
		.value("VERTEX", ShaderStageType::VERTEX)
		.value("FRAGMENT", ShaderStageType::FRAGMENT)
		.value("GEOMETRY", ShaderStageType::GEOMETRY)
		.value("TESS_CONTROL", ShaderStageType::TESS_CONTROL)
		.value("TESS_EVAL", ShaderStageType::TESS_EVAL)
		.value("COMPUTE", ShaderStageType::COMPUTE);

	py::enum_<DescriptorType>(m, "DescriptorType")
		.value("STORAGE_BUFFER", DescriptorType::STORAGE_BUFFER)
		.value("UNIFORM_BUFFER", DescriptorType::UNIFORM_BUFFER)
		.value("STORAGE_IMAGE", DescriptorType::STORAGE_IMAGE)
		.value("SAMPLED_IMAGE", DescriptorType::SAMPLED_IMAGE)
		.value("SAMPLER", DescriptorType::SAMPLER)
		.value("COMBINED_IMAGE_SAMPLER", DescriptorType::COMBINED_IMAGE_SAMPLER)
		.value("ACCELERATION_STRUCTURE", DescriptorType::ACCELERATION_STRUCTURE);

	py::enum_<Filter>(m, "Filter")
		.value("NEAREST", Filter::NEAREST)
		.value("LINEAR", Filter::LINEAR)
		.export_values();

	py::enum_<Format>(m, "Format")
		.value("Undefined", Format::Undefined)
		.value("R8_UNorm", Format::R8_UNorm).value("RG8_UNorm", Format::RG8_UNorm)
		.value("RGB8_UNorm", Format::RGB8_UNorm).value("RGBA8_UNorm", Format::RGBA8_UNorm)
		.value("BGRA8_UNorm", Format::BGRA8_UNorm)
		.value("R8_SNorm", Format::R8_SNorm).value("RG8_SNorm", Format::RG8_SNorm)
		.value("RGB8_SNorm", Format::RGB8_SNorm).value("RGBA8_SNorm", Format::RGBA8_SNorm)
		.value("R8_UInt", Format::R8_UInt).value("RG8_UInt", Format::RG8_UInt)
		.value("RGB8_UInt", Format::RGB8_UInt).value("RGBA8_UInt", Format::RGBA8_UInt)
		.value("R8_SInt", Format::R8_SInt).value("RG8_SInt", Format::RG8_SInt)
		.value("RGB8_SInt", Format::RGB8_SInt).value("RGBA8_SInt", Format::RGBA8_SInt)
		.value("R16_UNorm", Format::R16_UNorm).value("RG16_UNorm", Format::RG16_UNorm)
		.value("RGB16_UNorm", Format::RGB16_UNorm).value("RGBA16_UNorm", Format::RGBA16_UNorm)
		.value("R16_SNorm", Format::R16_SNorm).value("RG16_SNorm", Format::RG16_SNorm)
		.value("RGB16_SNorm", Format::RGB16_SNorm).value("RGBA16_SNorm", Format::RGBA16_SNorm)
		.value("R16_UInt", Format::R16_UInt).value("RG16_UInt", Format::RG16_UInt)
		.value("RGB16_UInt", Format::RGB16_UInt).value("RGBA16_UInt", Format::RGBA16_UInt)
		.value("R16_SInt", Format::R16_SInt).value("RG16_SInt", Format::RG16_SInt)
		.value("RGB16_SInt", Format::RGB16_SInt).value("RGBA16_SInt", Format::RGBA16_SInt)
		.value("R32_UInt", Format::R32_UInt).value("RG32_UInt", Format::RG32_UInt)
		.value("RGB32_UInt", Format::RGB32_UInt).value("RGBA32_UInt", Format::RGBA32_UInt)
		.value("R32_SInt", Format::R32_SInt).value("RG32_SInt", Format::RG32_SInt)
		.value("RGB32_SInt", Format::RGB32_SInt).value("RGBA32_SInt", Format::RGBA32_SInt)
		.value("R64_UInt", Format::R64_UInt).value("RG64_UInt", Format::RG64_UInt)
		.value("RGB64_UInt", Format::RGB64_UInt).value("RGBA64_UInt", Format::RGBA64_UInt)
		.value("R64_SInt", Format::R64_SInt).value("RG64_SInt", Format::RG64_SInt)
		.value("RGB64_SInt", Format::RGB64_SInt).value("RGBA64_SInt", Format::RGBA64_SInt)
		.value("R16_Float", Format::R16_Float).value("RG16_Float", Format::RG16_Float)
		.value("RGB16_Float", Format::RGB16_Float).value("RGBA16_Float", Format::RGBA16_Float)
		.value("R32_Float", Format::R32_Float).value("RG32_Float", Format::RG32_Float)
		.value("RGB32_Float", Format::RGB32_Float).value("RGBA32_Float", Format::RGBA32_Float)
		.value("R64_Float", Format::R64_Float).value("RG64_Float", Format::RG64_Float)
		.value("RGB64_Float", Format::RGB64_Float).value("RGBA64_Float", Format::RGBA64_Float);

	py::class_<Buffer, std::shared_ptr<Buffer>>(m, "Buffer")
		.def(
			"__dlpack__",
			// The DLPack protocol calls this with a `stream` keyword (a CUDA
			// stream pointer, for GPU tensors) that consumers are supposed to
			// synchronize against before reading the data. There's no stream
			// handling to do here (this isn't a CUDA-side export), so it's
			// accepted and ignored.
			[](const Buffer& self, py::object /*stream*/) { return self.vk_dlpack(); },
			py::arg("stream") = py::none()
		)
		.def(
			"__dlpack_device__",
			[](const Buffer& self) {
				DLDevice d = self.vk_dlpack_device();
				return py::make_tuple(d.device_type, d.device_id);
			}
		)
		.def("field", &Buffer::field, py::arg("field"))
		.def("read", &Buffer::read, py::arg("field"))
		.def(
			"write",
			[](const Buffer& self, const LayoutField& field, py::object value) {
				// Python-side convenience only: a plain list/tuple for a
				// VECTOR/MATRIX field is converted to an array.array before
				// reaching Buffer::write(), which itself only ever sees a
				// buffer-protocol/DLPack object, unchanged.
				const bool is_iterable = py::isinstance<py::list>(value) || py::isinstance<py::tuple>(value);
				const bool is_vec_or_matrix = field.layout
					&& (field.layout->kind == TypeKind::VECTOR || field.layout->kind == TypeKind::MATRIX);
				if (is_iterable && is_vec_or_matrix) {
					value = iterable_to_buffer(field.layout->component_type, value);
				}
				self.write(field, value);
			},
			py::arg("field"), py::arg("value")
		)
		.def("load", &Buffer::load, py::arg("source"))
		.def("save", &Buffer::save, py::arg("target"))
		.def(
			"cast",
			py::overload_cast<ScalarType>(&Buffer::cast, py::const_),
			py::arg("scalar")
		)
		.def(
			"cast",
			py::overload_cast<Format>(&Buffer::cast, py::const_),
			py::arg("format")
		)
		.def(
			"cast",
			py::overload_cast<const std::shared_ptr<Layout>&>(&Buffer::cast, py::const_),
			py::arg("layout")
		)
		.def("slice", &Buffer::slice, py::arg("start_element"), py::arg("count"))
		.def("element", &Buffer::element, py::arg("index"))
		.def_property_readonly("element_layout", &Buffer::element_layout)
		.def_property_readonly("size", &Buffer::size)
		.def_property_readonly("device_ptr", &Buffer::device_ptr);

	py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
		.def(
			"__dlpack__",
			[](const Tensor& self, py::object /*stream*/) { return self.vk_dlpack(); },
			py::arg("stream") = py::none()
		)
		.def(
			"__dlpack_device__",
			[](const Tensor& self) {
				DLDevice d = self.vk_dlpack_device();
				return py::make_tuple(d.device_type, d.device_id);
			}
		)
		.def("load", &Tensor::load, py::arg("source"))
		.def("save", &Tensor::save, py::arg("target"))
		.def_property_readonly("shape", &Tensor::shape)
		.def_property_readonly("scalar_type", &Tensor::scalar_type)
		.def_property_readonly("size", &Tensor::size)
		.def_property_readonly("device_ptr", &Tensor::device_ptr);

	py::class_<WrappedMemory, std::shared_ptr<WrappedMemory>>(m, "WrappedMemory")
		.def("unwrap", &WrappedMemory::unwrap)
		.def_property_readonly("device_ptr", &WrappedMemory::device_ptr)
		.def_property_readonly("shape", &WrappedMemory::shape)
		.def_property_readonly("scalar_type", &WrappedMemory::scalar_type);

	py::class_<Image, std::shared_ptr<Image>>(m, "Image")
		.def("cast_format", &Image::cast_format, py::arg("format"))
		.def("slice", &Image::slice, py::arg("mip_start"), py::arg("mip_count"), py::arg("array_start"), py::arg("array_count"))
		.def_property_readonly("format", &Image::format)
		.def_property_readonly("mip_count", &Image::mip_count)
		.def_property_readonly("array_count", &Image::array_count);

	py::class_<TypeDescriptor, std::shared_ptr<TypeDescriptor>>(m, "TypeDescriptor")
		.def_static("scalar", &TypeDescriptor::scalar, py::arg("type"))
		.def_static("vector", &TypeDescriptor::vector, py::arg("component_type"), py::arg("components"))
		.def_static("matrix", &TypeDescriptor::matrix, py::arg("component_type"), py::arg("rows"), py::arg("columns"))
		.def_static("array_of", &TypeDescriptor::array_of, py::arg("element_type"), py::arg("count"))
		.def_static(
			"struct_of",
			[](std::vector<std::pair<std::string, std::shared_ptr<TypeDescriptor>>> fields) {
				std::vector<StructField> struct_fields;
				struct_fields.reserve(fields.size());
				for (auto& field : fields)
					struct_fields.push_back(StructField{ field.first, field.second });
				return TypeDescriptor::struct_of(std::move(struct_fields));
			},
			py::arg("fields")
		)
		.def_property_readonly("kind", &TypeDescriptor::kind);

	py::class_<LayoutField>(m, "LayoutField")
		.def_readonly("name", &LayoutField::name)
		.def_readonly("offset", &LayoutField::offset)
		.def_readonly("layout", &LayoutField::layout)
		.def_property_readonly("root", [](const LayoutField& f) { return f.root.lock(); });

	py::class_<Layout, std::shared_ptr<Layout>>(m, "Layout")
		.def_readonly("size", &Layout::size)
		.def_readonly("alignment", &Layout::alignment)
		.def_readonly("aligned_size", &Layout::aligned_size)
		.def_readonly("kind", &Layout::kind)
		.def_readonly("component_type", &Layout::component_type)
		.def_readonly("fields", &Layout::fields)
		.def_readonly("element_layout", &Layout::element_layout)
		.def_readonly("stride", &Layout::stride)
		.def_readonly("count", &Layout::count);

	m.def(
		"compute_layout",
		&compute_layout,
		py::arg("type"),
		py::arg("rule"),
		"Computes the byte layout (size, alignment, offsets/strides) of a TypeDescriptor under a given LayoutRule."
	);

	py::class_<CommandBuffer, std::shared_ptr<CommandBuffer>>(m, "CommandBuffer")
		.def("close", &CommandBuffer::close)
		.def(
			"transfer",
			&CommandBuffer::transfer,
			py::arg("source"),
			py::arg("destination")
		)
		.def(
			"set_pipeline",
			&CommandBuffer::set_pipeline,
			py::arg("pipeline")
		)
		.def(
			"bind",
			&CommandBuffer::bind,
			py::arg("initial_set"),
			py::arg("descriptor_sets")
		)
		.def(
			"dispatch_threads",
			&CommandBuffer::dispatch_threads,
			py::arg("x"),
			py::arg("y") = 1,
			py::arg("z") = 1
		)
		.def(
			"set_framebuffer",
			&CommandBuffer::set_framebuffer,
			py::arg("framebuffer")
		)
		.def(
			"set_viewport",
			&CommandBuffer::set_viewport,
			py::arg("x"),
			py::arg("y"),
			py::arg("width"),
			py::arg("height")
		)
		.def(
			"blit_image",
			&CommandBuffer::blit_image,
			py::arg("src"),
			py::arg("dst"),
			py::arg("filter") = Filter::LINEAR
		)
		.def(
			"bind_vertices",
			&CommandBuffer::bind_vertices,
			py::arg("binding"),
			py::arg("vertex_buffer")
		)
		.def(
			"bind_indices",
			&CommandBuffer::bind_indices,
			py::arg("index_buffer")
		)
		.def(
			"dispatch_primitives",
			&CommandBuffer::dispatch_primitives,
			py::arg("vertices"),
			py::arg("vertex_start") = 0
		)
		.def(
			"dispatch_indexed_primitives",
			&CommandBuffer::dispatch_indexed_primitives,
			py::arg("indices"),
			py::arg("index_start") = 0,
			py::arg("vertex_offset") = 0
		)
		.def_property_readonly("is_submitted", &CommandBuffer::is_submitted)
		.def_property_readonly("is_executable", &CommandBuffer::is_executable)
		.def_property_readonly("is_closed", &CommandBuffer::is_closed)
		.def_property_readonly("is_released", &CommandBuffer::is_released);

	py::class_<SubmittedTask, std::shared_ptr<SubmittedTask>>(m, "SubmissionTask")
		.def("wait", &SubmittedTask::wait)
		.def_property_readonly("is_complete", &SubmittedTask::is_complete);

	py::class_<Engine, std::shared_ptr<Engine>>(m, "Engine")
		.def("create_command_buffer", &Engine::create_command_buffer)
		.def("submit", &Engine::submit, py::arg("command_buffers"))
		.def("wait", &Engine::wait);

	py::class_<ShaderSource>(m, "ShaderSource")
		.def_static(
			"from_spirv",
			&ShaderSource::from_spirv,
			py::arg("code"),
			py::arg("entry_point") = "main"
		)
		.def_static(
			"from_file",
			&ShaderSource::from_file,
			py::arg("path"),
			py::arg("stage"),
			py::arg("entry_point") = "main",
			py::arg("include_dirs") = std::vector<std::string>{}
		)
		.def_static(
			"from_glsl",
			&ShaderSource::from_glsl,
			py::arg("source"),
			py::arg("stage"),
			py::arg("entry_point") = "main",
			py::arg("include_dirs") = std::vector<std::string>{}
		);

	py::class_<Framebuffer, std::shared_ptr<Framebuffer>>(m, "Framebuffer")
		.def_property_readonly("width", &Framebuffer::width)
		.def_property_readonly("height", &Framebuffer::height);

	py::class_<Frame, std::shared_ptr<Frame>>(m, "Frame")
		.def("render_target", &Frame::render_target)
		.def("image_target", &Frame::image_target)
		.def("buffer_target", &Frame::buffer_target)
		.def("tensor_target", &Frame::tensor_target)
		.def_property_readonly("index", &Frame::index)
		.def_property_readonly("is_target_acquired", &Frame::is_target_acquired)
		.def_property_readonly("is_render_target", &Frame::is_render_target)
		.def_property_readonly("is_image_target", &Frame::is_image_target)
		.def_property_readonly("is_buffer_target", &Frame::is_buffer_target)
		.def_property_readonly("is_tensor_target", &Frame::is_tensor_target)
		.def_property_readonly("presented", &Frame::presented)
		.def("present", &Frame::present);

	py::class_<Stats, std::shared_ptr<Stats>>(m, "Stats")
		.def_property_readonly("fps", &Stats::fps);

	py::class_<Checkbox, std::shared_ptr<Checkbox>>(m, "Checkbox")
		.def("draw", &Checkbox::draw)
		.def_property("value", &Checkbox::value, &Checkbox::set_value);

	py::class_<SliderFloat, std::shared_ptr<SliderFloat>>(m, "SliderFloat")
		.def("draw", &SliderFloat::draw)
		.def_property("value", &SliderFloat::value, &SliderFloat::set_value);

	py::class_<SliderInt, std::shared_ptr<SliderInt>>(m, "SliderInt")
		.def("draw", &SliderInt::draw)
		.def_property("value", &SliderInt::value, &SliderInt::set_value);

	py::class_<Combobox, std::shared_ptr<Combobox>>(m, "Combobox")
		.def("draw", &Combobox::draw)
		.def_property("selected_index", &Combobox::selected_index, &Combobox::set_selected_index)
		.def_property_readonly("selected_item", &Combobox::selected_item)
		.def_property_readonly("items", &Combobox::items);

	py::class_<Window, std::shared_ptr<Window>>(m, "Window")
		.def("check_alive", &Window::check_alive)
		.def("set_title", &Window::set_title, py::arg("title"))
		.def("set_size", &Window::set_size, py::arg("width"), py::arg("height"))
		.def_property_readonly("width", &Window::width)
		.def_property_readonly("height", &Window::height)
		.def_property_readonly("stats", &Window::stats)
		.def("begin_frame", &Window::begin_frame, py::return_value_policy::move)
		.def(
			"label",
			static_cast<void (Window::*)(const std::string&)>(&Window::label),
			py::arg("text")
		)
		.def(
			"label",
			static_cast<void (Window::*)(const std::string&, double)>(&Window::label),
			py::arg("text"), py::arg("value")
		)
		.def(
			"label",
			static_cast<void (Window::*)(const std::string&, const std::string&)>(&Window::label),
			py::arg("text"), py::arg("value")
		)
		.def("button", &Window::button, py::arg("text"))
		.def("checkbox", &Window::checkbox, py::arg("label"), py::arg("value") = false)
		.def("slider_float", &Window::slider_float, py::arg("label"), py::arg("min"), py::arg("max"), py::arg("value"))
		.def("slider_int", &Window::slider_int, py::arg("label"), py::arg("min"), py::arg("max"), py::arg("value"))
		.def("combobox", &Window::combobox, py::arg("label"), py::arg("items"), py::arg("selected_index") = 0);

	// Opaque handles: not constructible from Python (no .def(py::init<...>())),
	// only obtainable from Pipeline::layout()/attach() and handed back to
	// DescriptorSet::bind()/Pipeline::create_framebuffer().
	py::class_<LayoutHandle>(m, "LayoutHandle");
	py::class_<AttachHandle>(m, "AttachHandle");

	py::class_<DescriptorSet, std::shared_ptr<DescriptorSet>>(m, "DescriptorSet")
		.def(
			"bind",
			py::overload_cast<LayoutHandle, const std::shared_ptr<Buffer>&>(&DescriptorSet::bind),
			py::arg("layout_id"),
			py::arg("buffer")
		)
		.def(
			"bind",
			py::overload_cast<LayoutHandle, const std::shared_ptr<Image>&>(&DescriptorSet::bind),
			py::arg("layout_id"),
			py::arg("image")
		);

	py::class_<Pipeline, std::shared_ptr<Pipeline>>(m, "Pipeline")
		.def("stage", &Pipeline::stage, py::arg("type"), py::arg("source"))
		.def(
			"layout",
			&Pipeline::layout,
			py::arg("set"),
			py::arg("binding"),
			py::arg("description"),
			py::arg("count") = 1
		)
		.def("vertex_layout", &Pipeline::vertex_layout, py::arg("start_location"), py::arg("layout"))
		.def("attach", &Pipeline::attach, py::arg("slot"), py::arg("format"))
		.def("local_size", &Pipeline::local_size, py::arg("x"), py::arg("y") = 1, py::arg("z") = 1)
		.def("close", &Pipeline::close)
		.def(
			"create_framebuffer",
			&Pipeline::create_framebuffer,
			py::arg("attachments"),
			py::return_value_policy::move
		)
		.def(
			"create_descriptor_set",
			&Pipeline::create_descriptor_set,
			py::arg("set") = 0,
			py::return_value_policy::move
		)
		.def_property_readonly("is_closed", &Pipeline::is_closed);

	py::class_<Device, std::shared_ptr<Device>>(m, "Device")
		.def("dispose", &Device::dispose)
		.def(
			"create_tensor",
			&Device::create_tensor,
			py::arg("shape"),
			py::arg("scalar_type"),
			py::arg("location"),
			py::return_value_policy::move
		)
		.def(
			"create_buffer",
			py::overload_cast<std::uint64_t, ScalarType, MemoryLocation>(&Device::create_buffer),
			py::arg("elements"),
			py::arg("scalar_type"),
			py::arg("location"),
			py::return_value_policy::move
		)
		.def(
			"create_buffer",
			py::overload_cast<std::uint64_t, Format, MemoryLocation>(&Device::create_buffer),
			py::arg("elements"),
			py::arg("format"),
			py::arg("location"),
			py::return_value_policy::move
		)
		.def(
			"create_buffer",
			py::overload_cast<std::uint64_t, const std::shared_ptr<Layout>&, MemoryLocation>(&Device::create_buffer),
			py::arg("elements"),
			py::arg("layout"),
			py::arg("location"),
			py::return_value_policy::move
		)
		.def(
			"create_image",
			&Device::create_image,
			py::arg("width"),
			py::arg("height"),
			py::arg("depth"),
			py::arg("mip_levels"),
			py::arg("array_layers"),
			py::arg("format"),
			py::arg("location"),
			py::return_value_policy::move
		)
		.def(
			"create_window",
			&Device::create_window,
			py::arg("width"),
			py::arg("height"),
			py::arg("title"),
			py::arg("format"),
			py::arg("frames_on_the_fly") = 3,
			py::return_value_policy::move
		)
		.def(
			"create_pipeline",
			&Device::create_pipeline,
			py::arg("type"),
			py::return_value_policy::move
		)
		.def(
			"create_engine",
			&Device::create_engine,
			py::arg("type"),
			py::arg("index") = 0,
			py::return_value_policy::move
		)
		.def(
			"create_staging",
			py::overload_cast<const std::shared_ptr<Buffer>&, MemoryLocation>(&Device::create_staging),
			py::arg("buffer"),
			py::arg("location") = MemoryLocation::HOST,
			py::return_value_policy::move
		)
		.def(
			"create_staging",
			py::overload_cast<const std::shared_ptr<Image>&, MemoryLocation>(&Device::create_staging),
			py::arg("image"),
			py::arg("location") = MemoryLocation::HOST,
			py::return_value_policy::move
		)
		.def(
			"wrap",
			&Device::wrap,
			py::arg("obj"),
			py::arg("mode"),
			py::arg("location") = MemoryLocation::DEVICE,
			py::return_value_policy::move
		)
		.def_static(
			"create_device",
			&Device::create_device,
			py::arg("device_index") = 0,
			py::arg("enable_validation_layers") = false,
			py::return_value_policy::move
		);

	m.def(
		"create_device",
		&Device::create_device,
		py::arg("device_index") = 0,
		py::arg("enable_validation_layers") = false,
		py::return_value_policy::move,
		"Convenience wrapper around Device.create_device"
	);
}

