#include "context.hpp"
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

namespace vk_context {

namespace {

std::map<std::uint32_t, std::shared_ptr<Device>> g_devices;
std::shared_ptr<Device> g_active_device;
std::map<std::uint32_t, std::shared_ptr<Engine>> g_active_engine;

bool env_flag_true(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) return false;
    std::string v(value);
    return v == "True" || v == "true" || v == "1";
}

} // namespace

std::shared_ptr<Device> resolve_device_index(std::uint32_t index) {
    auto it = g_devices.find(index);
    if (it != g_devices.end()) return it->second;
    bool use_debug = env_flag_true("VK_DEBUG");
    if (use_debug) {
        std::cout << "[INFO] Using validation layers." << std::endl;
    }
    auto dev = Device::create_device(index, use_debug);
    g_devices[index] = dev;
    return dev;
}

std::shared_ptr<Device> current_device() {
    if (!g_active_device) {
        g_active_device = resolve_device_index(0);
    }
    return g_active_device;
}

std::uint32_t current_device_index() {
    return current_device()->device_index();
}

Caps caps() {
    return current_device()->caps();
}

void DeviceContext::exit(const pybind11::object&, const pybind11::object&, const pybind11::object&) {
    g_active_device = previous_;
}

DeviceContext device(std::uint32_t device_index) {
    std::shared_ptr<Device> previous = g_active_device;
    g_active_device = resolve_device_index(device_index);
    return DeviceContext(previous);
}

EngineType default_engine_type() {
    return static_cast<EngineType>(
        static_cast<int>(EngineType::GRAPHICS) | static_cast<int>(EngineType::COMPUTE));
}

void EngineContext::exit(const pybind11::object&, const pybind11::object&, const pybind11::object&) {
    g_active_engine[device_index_] = previous_;
}

EngineContext engine(std::optional<EngineType> engine_type, std::uint32_t engine_index) {
    auto dev = current_device();
    std::uint32_t di = dev->device_index();
    std::shared_ptr<Engine> previous = g_active_engine[di];
    EngineType requested = engine_type.has_value() ? *engine_type : default_engine_type();
    if (!previous || previous->engine_type() != requested) {
        g_active_engine[di] = dev->create_engine(requested, engine_index);
    }
    return EngineContext(di, previous);
}

std::shared_ptr<Engine> current_engine() {
    auto dev = current_device();
    std::uint32_t di = dev->device_index();
    auto it = g_active_engine.find(di);
    if (it == g_active_engine.end() || !it->second) {
        engine();
        it = g_active_engine.find(di);
    }
    return it->second;
}

void dispose() {
    if (!g_active_device) return;
    auto dev = g_active_device;
    std::uint32_t di = dev->device_index();
    g_devices.erase(di);
    g_active_engine.erase(di);
    g_active_device.reset();
    dev->dispose();
}

void relax() {
    g_active_device.reset();
    g_devices.clear();
    g_active_engine.clear();
}

std::shared_ptr<Tensor> tensor(const std::vector<std::uint64_t>& shape, Type scalar_type, MemoryLocation location) {
    return current_device()->create_tensor(shape, scalar_type, location);
}

std::shared_ptr<Buffer> buffer_of_type(std::uint64_t elements, Type element_type, MemoryLocation location) {
    return current_device()->create_buffer(elements, element_type, location);
}

std::shared_ptr<Buffer> buffer_of_format(std::uint64_t elements, Format format, MemoryLocation location) {
    return current_device()->create_buffer(elements, format, location);
}

std::shared_ptr<Buffer> buffer_of_layout(std::uint64_t elements, const std::shared_ptr<Layout>& layout, MemoryLocation location) {
    return current_device()->create_buffer(elements, layout, location);
}

std::shared_ptr<Image> image(
    int width, int height, int depth, int mip_levels, int array_layers, Format format, MemoryLocation location) {
    return current_device()->create_image(width, height, depth, mip_levels, array_layers, format, location);
}

std::shared_ptr<Image> depth_buffer_image(int width, int height, Format format, MemoryLocation location) {
    return current_device()->create_depth_buffer_image(width, height, format, location);
}

std::shared_ptr<Sampler> sampler(
    Filter mag_filter, Filter min_filter, MipmapMode mipmap_mode,
    WrapMode wrap_u, WrapMode wrap_v, WrapMode wrap_w) {
    return current_device()->create_sampler(mag_filter, min_filter, mipmap_mode, wrap_u, wrap_v, wrap_w);
}

std::shared_ptr<AccelerationStructure> ads(const ADSDeclaration& declaration) {
    return current_device()->create_ads(declaration);
}

std::shared_ptr<Window> window(
    std::uint32_t width, std::uint32_t height, const std::string& title, Format format,
    std::uint32_t frames_on_the_fly, bool vsync) {
    return current_device()->create_window(width, height, title, format, frames_on_the_fly, vsync);
}

std::shared_ptr<Buffer> staging_for_buffer(const std::shared_ptr<Buffer>& buffer, MemoryLocation location) {
    return current_device()->create_staging(buffer, location);
}

std::shared_ptr<Buffer> staging_for_image(const std::shared_ptr<Image>& image, MemoryLocation location) {
    return current_device()->create_staging(image, location);
}

std::shared_ptr<Pipeline> pipeline(PipelineType type) {
    return current_device()->create_pipeline(type);
}

std::shared_ptr<WrappedMemory> wrap(pybind11::object obj, MemoryLocation location) {
    return current_device()->wrap(std::move(obj), location);
}

std::shared_ptr<Scene> load_scene(const std::string& filename, VertexResolutionMode resolution_mode) {
    return ::load_scene(current_device(), filename, resolution_mode);
}

std::shared_ptr<CommandBuffer> command_buffer() {
    return current_engine()->create_command_buffer();
}

std::shared_ptr<SubmittedTask> submit(std::vector<std::shared_ptr<CommandBuffer>> command_buffers) {
    return current_engine()->submit(std::move(command_buffers));
}

void wait() {
    current_engine()->wait();
}

void RecordingContext::exit(const pybind11::object&, const pybind11::object&, const pybind11::object&) {
    cmd_->close();
    engine_->submit({cmd_})->wait();
}

namespace {

RecordingContext make_recording_context(EngineType type, std::uint32_t engine_index) {
    auto eng = current_device()->create_engine(type, engine_index);
    auto cmd = eng->create_command_buffer();
    return RecordingContext(eng, cmd);
}

} // namespace

RecordingContext transfer(std::uint32_t engine_index) {
    return make_recording_context(EngineType::TRANSFER, engine_index);
}

RecordingContext compute(std::uint32_t engine_index) {
    return make_recording_context(
        static_cast<EngineType>(static_cast<int>(EngineType::COMPUTE) | static_cast<int>(EngineType::TRANSFER)),
        engine_index);
}

RecordingContext graphics(std::uint32_t engine_index) {
    return make_recording_context(
        static_cast<EngineType>(
            static_cast<int>(EngineType::GRAPHICS) | static_cast<int>(EngineType::COMPUTE) | static_cast<int>(EngineType::TRANSFER)),
        engine_index);
}

} // namespace vk_context
