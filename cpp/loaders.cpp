//
// Created by mendez on 7/15/2026.
//

#include <cctype>
#include <cstring>
#include <iostream>
#include <map>
#include <unordered_map>

#include <tiny_obj_loader.h>

#include "device.hpp"

namespace {

// tinyobj always fills these arrays (defaulting to 0), so every property is
// set unconditionally except the texture maps, which tinyobj leaves empty
// when absent.
std::shared_ptr<Material> convert_material(const tinyobj::material_t& src) {
    auto material = std::make_shared<Material>();
    material->set("name", src.name);
    material->set("ambient", std::array<float, 3>{src.ambient[0], src.ambient[1], src.ambient[2]});
    material->set("diffuse", std::array<float, 3>{src.diffuse[0], src.diffuse[1], src.diffuse[2]});
    material->set("specular", std::array<float, 3>{src.specular[0], src.specular[1], src.specular[2]});
    material->set("transmittance", std::array<float, 3>{src.transmittance[0], src.transmittance[1], src.transmittance[2]});
    material->set("emission", std::array<float, 3>{src.emission[0], src.emission[1], src.emission[2]});
    material->set("shininess", src.shininess);
    material->set("ior", src.ior);
    material->set("dissolve", src.dissolve);
    if (!src.diffuse_texname.empty()) material->set("diffuse_map", src.diffuse_texname);
    if (!src.specular_texname.empty()) material->set("specular_map", src.specular_texname);
    if (!src.normal_texname.empty()) material->set("normal_map", src.normal_texname);
    if (!src.bump_texname.empty()) material->set("bump_map", src.bump_texname);
    if (!src.alpha_texname.empty()) material->set("alpha_map", src.alpha_texname);
    return material;
}

// tinyobj::index_t (vertex/normal/texcoord index triple) has no operator==
// or hash; wrapped here so a resolution key can be hashed. A component set
// to -1 means "not part of the key" (either the attribute doesn't exist in
// the file, or VertexResolutionMode::ByPosition ignores it).
struct VertexKey {
    int v, vn, vt;
    bool operator==(const VertexKey& o) const noexcept { return v == o.v && vn == o.vn && vt == o.vt; }
};
struct VertexKeyHash {
    size_t operator()(const VertexKey& k) const noexcept {
        size_t h = std::hash<int>{}(k.v);
        h = h * 31 + std::hash<int>{}(k.vn);
        h = h * 31 + std::hash<int>{}(k.vt);
        return h;
    }
};

// Accumulates one material group's worth of welded vertices/indices.
struct VertexGroup {
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    std::unordered_map<VertexKey, uint32_t, VertexKeyHash> unique_vertices;
};

// Appends `idx`'s vertex data to `group.vertices` and returns its output
// index, welding it against a prior identical vertex per `mode` (see
// VertexResolutionMode).
uint32_t resolve_vertex(
    VertexGroup& group, const tinyobj::attrib_t& attrib, const tinyobj::index_t& idx,
    bool has_normals, bool has_texcoords, size_t stride, VertexResolutionMode mode)
{
    auto push_new = [&]() -> uint32_t {
        const auto vertex_index = static_cast<uint32_t>(group.vertices.size() / stride);
        group.vertices.push_back(attrib.vertices[3 * idx.vertex_index + 0]);
        group.vertices.push_back(attrib.vertices[3 * idx.vertex_index + 1]);
        group.vertices.push_back(attrib.vertices[3 * idx.vertex_index + 2]);
        if (has_normals) {
            group.vertices.push_back(attrib.normals[3 * idx.normal_index + 0]);
            group.vertices.push_back(attrib.normals[3 * idx.normal_index + 1]);
            group.vertices.push_back(attrib.normals[3 * idx.normal_index + 2]);
        }
        if (has_texcoords) {
            group.vertices.push_back(attrib.texcoords[2 * idx.texcoord_index + 0]);
            group.vertices.push_back(attrib.texcoords[2 * idx.texcoord_index + 1]);
        }
        return vertex_index;
    };

    if (mode == VertexResolutionMode::Duplicate) return push_new();

    const bool by_position_only = mode == VertexResolutionMode::ByPosition;
    const VertexKey key{
        idx.vertex_index,
        (!by_position_only && has_normals) ? idx.normal_index : -1,
        (!by_position_only && has_texcoords) ? idx.texcoord_index : -1,
    };
    auto found = group.unique_vertices.find(key);
    if (found != group.unique_vertices.end()) return found->second;
    const uint32_t vertex_index = push_new();
    group.unique_vertices.emplace(key, vertex_index);
    return vertex_index;
}

// Uploads `data` into a new HOST-visible Buffer (mapped eagerly, so its
// external_ptr() is immediately writable) -- simplest possible upload path
// for scene loading, at the cost of not being DEVICE-local.
template<typename T>
std::shared_ptr<Buffer> upload(const std::shared_ptr<Device>& device, const std::vector<T>& data, Type scalar) {
    auto buffer = device->create_buffer(data.size(), scalar, MemoryLocation::HOST);
    std::memcpy(reinterpret_cast<void*>(buffer->external_ptr()), data.data(), data.size() * sizeof(T));
    return buffer;
}

std::string material_name_or(const std::shared_ptr<Material>& material, const std::string& fallback) {
    if (!material) return fallback;
    const MaterialValue* name = material->find("name");
    if (const auto* s = name ? std::get_if<std::string>(name) : nullptr; s && !s->empty()) return *s;
    return fallback;
}

std::shared_ptr<Scene> load_obj_scene(
    const std::shared_ptr<Device>& device, const std::string& filename, VertexResolutionMode resolution_mode)
{
    const auto slash = filename.find_last_of("/\\");
    const std::string base_dir = slash == std::string::npos ? "" : filename.substr(0, slash + 1);

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    bool ok = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str(), base_dir.c_str());
    if (!warn.empty()) std::cerr << "[load_scene] " << filename << ": " << warn << std::endl;
    if (!ok) throw std::runtime_error("load_scene: failed to load '" + filename + "': " + err);

    std::vector<std::shared_ptr<Material>> converted_materials;
    converted_materials.reserve(materials.size());
    for (const auto& m : materials) converted_materials.push_back(convert_material(m));

    const bool has_normals = !attrib.normals.empty();
    const bool has_texcoords = !attrib.texcoords.empty();
    std::vector<VertexAttribute> attributes{VertexAttribute::POSITION};
    if (has_normals) attributes.push_back(VertexAttribute::NORMAL);
    if (has_texcoords) attributes.push_back(VertexAttribute::TEXCOORD);
    const size_t stride = 3 + (has_normals ? 3 : 0) + (has_texcoords ? 2 : 0);

    auto scene = std::make_shared<Scene>();

    for (const auto& shape : shapes) {
        // Grouped by material id (-1 = no material) so a shape referencing
        // several materials produces one Mesh/SceneNode per material
        // instead of one node with a single, arbitrarily-chosen material.
        std::map<int, VertexGroup> groups;

        size_t index_offset = 0;
        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
            const int face_vertices = shape.mesh.num_face_vertices[f];
            const int material_id = shape.mesh.material_ids.empty() ? -1 : shape.mesh.material_ids[f];
            auto& group = groups[material_id];
            for (int v = 0; v < face_vertices; ++v) {
                const tinyobj::index_t& idx = shape.mesh.indices[index_offset + v];
                group.indices.push_back(resolve_vertex(group, attrib, idx, has_normals, has_texcoords, stride, resolution_mode));
            }
            index_offset += face_vertices;
        }

        const bool splits_into_multiple_nodes = groups.size() > 1;
        for (auto& [material_id, group] : groups) {
            if (group.indices.empty()) continue;

            auto mesh = std::make_shared<Mesh>(
                upload(device, group.vertices, Type::FLOAT32),
                upload(device, group.indices, Type::UINT32),
                attributes);

            std::shared_ptr<Material> material;
            if (material_id >= 0 && static_cast<size_t>(material_id) < converted_materials.size()) {
                material = converted_materials[material_id];
            }

            std::string node_name = shape.name;
            if (splits_into_multiple_nodes) {
                node_name += "_" + material_name_or(material, "mat" + std::to_string(material_id));
            }

            // OBJ has no per-node transform concept: nodes are left untransformed.
            scene->add_node(SceneNode(node_name, material, mesh, nullptr));
        }
    }

    return scene;
}

} // namespace

std::shared_ptr<Scene> load_scene(
    const std::shared_ptr<Device>& device, const std::string& filename, VertexResolutionMode resolution_mode)
{
    const auto dot = filename.find_last_of('.');
    std::string ext = dot == std::string::npos ? "" : filename.substr(dot + 1);
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (ext == "obj") return load_obj_scene(device, filename, resolution_mode);
    throw std::runtime_error("load_scene: unsupported file extension '" + ext + "'");
}
