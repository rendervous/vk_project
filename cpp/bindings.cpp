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

PYBIND11_MODULE(vk, m) {
	m.doc() = "Minimal Vulkan bindings";

	py::enum_<MemoryLocation>(m, "MemoryLocation")
		.value("HOST", MemoryLocation::HOST)
		.value("DEVICE", MemoryLocation::DEVICE)
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

	py::enum_<Format>(m, "Format")
		.value("Undefined", Format::Undefined)
		.value("R8_UNorm", Format::R8_UNorm).value("RG8_UNorm", Format::RG8_UNorm)
		.value("RGB8_UNorm", Format::RGB8_UNorm).value("RGBA8_UNorm", Format::RGBA8_UNorm)
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
		.def_property_readonly("size", &Buffer::size);

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

	py::class_<DescriptorSet, std::shared_ptr<DescriptorSet>>(m, "DescriptorSet")
		.def(
			"bind",
			py::overload_cast<int, const std::shared_ptr<Buffer>&>(&DescriptorSet::bind),
			py::arg("layout_id"),
			py::arg("buffer")
		)
		.def(
			"bind",
			py::overload_cast<int, const std::shared_ptr<Image>&>(&DescriptorSet::bind),
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
			"allocate_tensor_dlpack",
			&Device::create_tensor_dlpack,
			py::arg("shape"),
			py::arg("scalar_type"),
			py::arg("location"),
			py::return_value_policy::move
		)
		.def(
			"create_buffer",
			&Device::create_buffer,
			py::arg("size"),
			py::arg("location"),
			py::return_value_policy::move
		)
		.def(
			"create_array",
			&Device::create_array,
			py::arg("elements"),
			py::arg("scalar_type"),
			py::arg("location"),
			py::return_value_policy::move
		)
		.def(
			"create_structured_buffer",
			&Device::create_structured_buffer,
			py::arg("layout"),
			py::arg("location"),
			py::arg("count") = 1,
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

