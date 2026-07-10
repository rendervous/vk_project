#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "device.hpp"
#include <memory>

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

	py::class_<Buffer, std::shared_ptr<Buffer>>(m, "Buffer")
		.def("dlpack", &Buffer::dlpack);

	py::class_<CommandBuffer, std::shared_ptr<CommandBuffer>>(m, "CommandBuffer")
		.def("close", &CommandBuffer::close)
		.def(
			"transfer",
			&CommandBuffer::transfer,
			py::arg("source"),
			py::arg("destination"),
			py::arg("bytes")
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

	py::class_<Device, std::shared_ptr<Device>>(m, "Device")
		.def(py::init<>())
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
			"create_engine",
			&Device::create_engine,
			py::arg("type"),
			py::arg("index") = 0,
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

