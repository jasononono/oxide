#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION


#include "backend.hpp"
#include "typeutil.hpp"
#include "tensor.hpp"

#include <fstream>
#include <Metal/Metal.hpp>


namespace oxide {


    Metal::~Metal() {
        device->release();
        queue->release();
    }


    Shader::~Shader() {
        library->release();
        for (const std::pair<std::string, MTL::ComputePipelineState*>& pair : pipeline) {
            pair.second->release();
        }
    }


    TensorMemory::TensorMemory(): tensor_type(typeid(void)) {}

    TensorMemory::TensorMemory(void* _address, std::type_index _tensor_type): address(_address), tensor_type(_tensor_type) {}

    bool TensorMemory::operator==(const TensorMemory& other) const {
        return address == other.address && tensor_type == other.tensor_type;
    }

    bool TensorMemory::valid() const {
        return address == nullptr;
    }


    std::size_t TensorMemoryHash::operator()(const TensorMemory &x) const {
        return std::hash<void*>()(x.address) ^ std::hash<std::type_index>()(x.tensor_type);
    }


    Backend::Backend():
    error(nullptr) {
        
        init_metal();
        
        random.generator =  std::mt19937(random.device());

        init_shader("src/shader.metal");
    }

    Backend::~Backend() {}

    void Backend::init_metal() {
        metal.device = MTL::CreateSystemDefaultDevice();
        if (!metal.device) {
            log("Oxide: failed to obtain gpu information."); abort();
        }
        metal.queue = metal.device->newCommandQueue();
        if (!metal.queue) {
            log("Oxide: failed to initialize gpu command queue."); abort();
        }
    }

    void Backend::init_shader(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            log("Oxide: shader source not found"); abort();
        }

        std::string text, line;
        while (getline(file, line)) {text += line + '\n';}
        file.close();

        NS::String* source = NS::String::string(text.c_str(), NS::UTF8StringEncoding);

        shader.library = metal.device->newLibrary(source, nullptr, &error);
        if (!shader.library) {
            log("Oxide: failed to compile shader."); log_metal(); abort();
        }
    }

    void Backend::load_shader(const std::string& name) {
        MTL::Function* function = shader.library->newFunction(NS::String::string(name.c_str(), NS::UTF8StringEncoding));
        if (!function) {
            log("Oxide: shader function '" + name + "' does not exist."); abort();
        }

        shader.pipeline[name] = metal.device->newComputePipelineState(function, &error);
        if (!shader.pipeline[name]) {
            log("Oxide: failed to create pipeline for '" + name + "'."); log_metal(); abort();
        }

        function->release();
    }

    NS::UInteger Backend::set_cps(MTL::ComputeCommandEncoder* encoder, const std::string& name) {
        if (!shader.pipeline.count(name)) {
            load_shader(name);
        }
        encoder->setComputePipelineState(shader.pipeline.at(name));
        return shader.pipeline.at(name)->maxTotalThreadsPerThreadgroup();
    }

    MTL::Buffer* Backend::new_buffer(int size) {
        return metal.device->newBuffer(size * sizeof(float), MTL::ResourceStorageModeShared);
    }

    MTL::CommandBuffer* Backend::new_cmd_buffer() {
        MTL::CommandBuffer* cmd = metal.queue->commandBuffer();
        if (!cmd) {
            log("Oxide: failed to initialize command buffer."); abort();
        }
        return cmd;
    }

    NS::Error** Backend::error_out() {
        return &error;
    }

    void Backend::log(const std::string& message) {
        error_log += message + '\n';
    }

    void Backend::log_metal() {
        if (error) {
            error_log += error->localizedDescription()->utf8String();
        }
    }

    void Backend::abort() {
        throw std::runtime_error(error_log);
    }

    std::mt19937& Backend::random_generate() {
        return random.generator;
    }

    TensorMemory Backend::memory_register(void* address, std::type_index tensor_type) {
        TensorMemory tensor_memory(address, tensor_type);
        memory.registered[tensor_memory] = std::unordered_set<TensorMemory, TensorMemoryHash>();
        memory.tensors.push_back(tensor_memory);
        return tensor_memory;
    }

    TensorMemory Backend::memory_register(TensorMemory parent_memory, void* address, std::type_index tensor_type) {
        TensorMemory tensor_memory(address, tensor_type);
        memory.registered[parent_memory].insert(tensor_memory);
        return tensor_memory;
    }

    void Backend::memory_unregister(TensorMemory parent_memory, TensorMemory view_memory) {
        memory.registered[parent_memory].erase(view_memory);
    }

    const std::vector<TensorMemory>& Backend::get_tensors() const {
        return memory.tensors;
    }

    const std::unordered_set<TensorMemory, TensorMemoryHash>& Backend::memory_tied(TensorMemory key) const {
        return memory.registered.at(key);
    }

    void Backend::memory_delete(TensorMemory tensor_memory) {
        memory.registered.erase(tensor_memory);
        memory.tensors.erase(std::find(memory.tensors.begin(), memory.tensors.end(), tensor_memory));
    }


}