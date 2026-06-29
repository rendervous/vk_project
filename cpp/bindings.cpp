#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "device.hpp"
#include <memory>

namespace py = pybind11;

PYBIND11_MODULE(_vk, m) {
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

	py::class_<Device, std::shared_ptr<Device>>(m, "Device")
		.def(py::init<>())
		.def("dispose", &Device::dispose)
		.def(
			"allocate_tensor_dlpack",
			&Device::allocate_tensor_dlpack,
			py::arg("shape"),
			py::arg("scalar_type"),
			py::arg("location"),
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

