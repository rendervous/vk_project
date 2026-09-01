#pragma once
// "Current device"/"current engine" context management, native (C++) side.
//
// Owns the state _context.py used to keep in module-level Python globals
// (the device registry keyed by physical device index, the active device,
// the active engine per device index) and every "shallow wrapper" free
// function that used to resolve __current_device()/__current_engine()
// itself before delegating to a Device/Engine method - device()/engine()
// now map directly to these, and tensor()/buffer()/image()/pipeline()/etc.
// are real C++ functions, not a Python indirection over Device's own
// methods.
#include "device.hpp"
#include <pybind11/pybind11.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace vk_context {

// ---- Registry / "current" state ----

// Looks `index` up in the process-wide device registry, creating (and
// registering) a new Device for it via Device::create_device() the first
// time it's requested. Validation layers are enabled iff the VK_DEBUG
// environment variable is set to "True"/"true"/"1" at that moment.
std::shared_ptr<Device> resolve_device_index(std::uint32_t index);

// The active device, activating device 0 first (via resolve_device_index)
// if device() was never called.
std::shared_ptr<Device> current_device();
std::uint32_t current_device_index();
// Optional-feature snapshot of the current device -- see Caps.
Caps caps();

// Returned by device(): restores the previously active device on __exit__
// (used as a context manager via `with`); discarding the return value (no
// `with`) makes the switch permanent.
class DeviceContext {
public:
    explicit DeviceContext(std::shared_ptr<Device> previous) : previous_(std::move(previous)) {}
    DeviceContext& enter() { return *this; }
    void exit(const pybind11::object& exc_type, const pybind11::object& exc_val, const pybind11::object& exc_tb);
private:
    std::shared_ptr<Device> previous_;
};

// Creates/activates the device at `device_index` as the current device.
DeviceContext device(std::uint32_t device_index = 0);

// GRAPHICS|COMPUTE: the default engine() tries first, falling back to a
// plain COMPUTE engine if this device has no queue family supporting both.
EngineType default_engine_type();

// Returned by engine(): restores the previously active engine (for this
// device) on __exit__; discarding the return value makes the switch
// permanent, same as device()/DeviceContext.
class EngineContext {
public:
    EngineContext(std::uint32_t device_index, std::shared_ptr<Engine> previous)
        : device_index_(device_index), previous_(std::move(previous)) {}
    EngineContext& enter() { return *this; }
    void exit(const pybind11::object& exc_type, const pybind11::object& exc_val, const pybind11::object& exc_tb);
private:
    std::uint32_t device_index_;
    std::shared_ptr<Engine> previous_;
};

// Creates/activates an engine on the current device as the current engine.
// `engine_type` = std::nullopt means "try GRAPHICS|COMPUTE, falling back to
// plain COMPUTE" (see default_engine_type()).
EngineContext engine(std::optional<EngineType> engine_type = std::nullopt, std::uint32_t engine_index = 0);

// The active engine for the current device, activating the default
// GRAPHICS|COMPUTE-or-COMPUTE engine first (via engine()) if engine() was
// never called for this device.
std::shared_ptr<Engine> current_engine();

// Disposes the current device, drops it from the registry (so a further
// device(same_index) call creates a fresh one), and clears the
// current-engine bookkeeping kept for it.
void dispose();

// Drops every reference this module keeps to created devices/engines
// (registry + active device/engine pointers), without disposing them - a
// device/engine still referenced elsewhere (a live Buffer/Image/
// CommandBuffer/etc.) stays alive via normal shared_ptr refcounting.
void relax();

// ---- Shallow wrappers over current_device()/current_engine() ----

std::shared_ptr<Tensor> tensor(const std::vector<std::uint64_t>& shape, Type scalar_type, MemoryLocation location = MemoryLocation::DEVICE);

std::shared_ptr<Buffer> buffer_of_type(std::uint64_t elements, Type element_type, MemoryLocation location = MemoryLocation::DEVICE);
std::shared_ptr<Buffer> buffer_of_format(std::uint64_t elements, Format format, MemoryLocation location = MemoryLocation::DEVICE);
std::shared_ptr<Buffer> buffer_of_layout(std::uint64_t elements, const std::shared_ptr<Layout>& layout, MemoryLocation location = MemoryLocation::DEVICE);

std::shared_ptr<Image> image(
    int width, int height = 1, int depth = 1, int mip_levels = 1, int array_layers = 1,
    Format format = Format::RGBA8_UNorm, MemoryLocation location = MemoryLocation::DEVICE);
std::shared_ptr<Image> depth_buffer_image(
    int width, int height, Format format = Format::Depth32_Float, MemoryLocation location = MemoryLocation::DEVICE);
std::shared_ptr<Sampler> sampler(
    Filter mag_filter = Filter::LINEAR,
    Filter min_filter = Filter::LINEAR,
    MipmapMode mipmap_mode = MipmapMode::LINEAR,
    WrapMode wrap_u = WrapMode::REPEAT,
    WrapMode wrap_v = WrapMode::REPEAT,
    WrapMode wrap_w = WrapMode::REPEAT);
std::shared_ptr<AccelerationStructure> ads(const ADSDeclaration& declaration);
std::shared_ptr<Window> window(
    std::uint32_t width, std::uint32_t height, const std::string& title, Format format,
    std::uint32_t frames_on_the_fly = 3, bool vsync = true);
std::shared_ptr<Buffer> staging_for_buffer(const std::shared_ptr<Buffer>& buffer, MemoryLocation location = MemoryLocation::HOST);
std::shared_ptr<Buffer> staging_for_image(const std::shared_ptr<Image>& image, MemoryLocation location = MemoryLocation::HOST);
std::shared_ptr<Pipeline> pipeline(PipelineType type);
std::shared_ptr<WrappedMemory> wrap(pybind11::object obj, MemoryLocation location = MemoryLocation::DEVICE);
std::shared_ptr<Scene> load_scene(const std::string& filename, VertexResolutionMode resolution_mode = VertexResolutionMode::ByAllAttributes);

std::shared_ptr<CommandBuffer> command_buffer();
std::shared_ptr<SubmittedTask> submit(std::vector<std::shared_ptr<CommandBuffer>> command_buffers);
void wait();

// transfer()/compute()/graphics(): a temporary command buffer on a
// TRANSFER / COMPUTE|TRANSFER / GRAPHICS|COMPUTE|TRANSFER engine,
// recording on __enter__, closed+submitted+waited on __exit__.
class RecordingContext {
public:
    RecordingContext(std::shared_ptr<Engine> engine, std::shared_ptr<CommandBuffer> cmd)
        : engine_(std::move(engine)), cmd_(std::move(cmd)) {}
    std::shared_ptr<CommandBuffer> enter() { return cmd_; }
    void exit(const pybind11::object& exc_type, const pybind11::object& exc_val, const pybind11::object& exc_tb);
private:
    std::shared_ptr<Engine> engine_;
    std::shared_ptr<CommandBuffer> cmd_;
};

RecordingContext transfer(std::uint32_t engine_index = 0);
RecordingContext compute(std::uint32_t engine_index = 0);
RecordingContext graphics(std::uint32_t engine_index = 0);

} // namespace vk_context
