#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION // TODO: undef this future me when you figure out where to do that
// i love talking with myself :D

#include "backend.hpp"
#include "common.hpp"
#include "tensor.hpp"


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
        return address != nullptr;
    }


    std::size_t TensorMemoryHash::operator()(const TensorMemory &x) const {
        return std::hash<void*>()(x.address) ^ std::hash<std::type_index>()(x.tensor_type);
    }


    Backend::Backend():
    mtl_err(nullptr) {
        
        init_metal();
        
        random.generator =  std::mt19937(random.device());
        random.seed_dist = std::uniform_int_distribution<uint>(0, std::numeric_limits<uint>::max());

        init_shader("src/shader.metal");
    }

    Backend::~Backend() {}

    void Backend::init_metal() {
        metal.device = MTL::CreateSystemDefaultDevice();
        if (!metal.device) {
            log("failed to obtain gpu information"); abort();
        }
        metal.queue = metal.device->newCommandQueue();
        if (!metal.queue) {
            log("failed to initialize gpu command queue");
            abort();
        }
    }

    void Backend::init_shader(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            log("shader source not found");
            abort();
        }

        std::string text, line;
        while (getline(file, line)) {text += line + '\n';}
        file.close();

        NS::String* source = NS::String::string(text.c_str(), NS::UTF8StringEncoding);

        shader.library = metal.device->newLibrary(source, nullptr, &mtl_err);
        if (!shader.library) {
            log("failed to compile shader\n");
            log_metal();
            abort();
        }
    }

    void Backend::load_shader(const std::string& name) {
        MTL::Function* function = shader.library->newFunction(NS::String::string(name.c_str(), NS::UTF8StringEncoding));
        if (!function) {
            log("shader function '" + name + "' does not exist");
            abort();
        }

        shader.pipeline[name] = metal.device->newComputePipelineState(function, &mtl_err);
        if (!shader.pipeline[name]) {
            log("failed to create pipeline for '" + name + "'"); log_metal();
            abort();
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

    MTL::Buffer* Backend::new_buffer(uint size) {
        return metal.device->newBuffer(size * sizeof(float), MTL::ResourceStorageModeShared);
    }

    MTL::CommandBuffer* Backend::new_cmd_buffer() {
        MTL::CommandBuffer* cmd = metal.queue->commandBuffer();
        if (!cmd) {
            log("failed to initialize command buffer"); abort();
        }
        return cmd;
    }

    NS::Error** Backend::get_mtl_err() {
        return &mtl_err;
    }

    void Backend::out(const std::string& message) {
        std::cout << "(!) " << message << std::endl;
    }

    void Backend::log(const std::string& message) {
        error_log += message + '\n';
    }

    void Backend::log_metal() {
        if (mtl_err) {
            error_log += mtl_err->localizedDescription()->utf8String();
        }
    }

    void Backend::abort() {
        throw oxide_error(error_log);
    }

    std::mt19937& Backend::random_generate() {
        return random.generator;
    }

    uint Backend::random_seed() {
        return random.seed_dist(random.generator);
    }

    TensorMemory Backend::mem_register(void* address, std::type_index tensor_type) {
        TensorMemory mem(address, tensor_type);
        memory.registered[mem] = std::unordered_set<TensorMemory, TensorMemoryHash>();
        memory.tensors.push_back(mem);
        return mem;
    }

    TensorMemory Backend::mem_register(TensorMemory parent_mem, void* address, std::type_index tensor_type) {
        TensorMemory mem(address, tensor_type);
        memory.registered[parent_mem].insert(mem);
        return mem;
    }

    void Backend::mem_unregister(TensorMemory parent_mem, TensorMemory view_mem) {
        if (!memory.registered.count(parent_mem)) {return;}
        memory.registered[parent_mem].erase(view_mem);

        if (memory.registered[parent_mem].empty()) {
            #define TEMPLATE(dtype) \
            if (parent_mem.tensor_type == typeid(TensorData<dtype>)) { \
                TensorData<dtype>* data = reinterpret_cast<TensorData<dtype>*>(parent_mem.address); \
                memory.cache += sizeof(*data) + data->get_size() * sizeof(dtype); \
                return; \
            }
            #include "specialize/all.h"

            log("tensor type is not recognized in memory, cannot add to cache");
            abort();
        }
    }

    const std::vector<TensorMemory>& Backend::get_tensors() const {
        return memory.tensors;
    }

    const std::unordered_set<TensorMemory, TensorMemoryHash>& Backend::get_mem_tied(TensorMemory key) const {
        return memory.registered.at(key);
    }

    void Backend::mem_delete(TensorMemory mem) {
        for (TensorMemory view_mem : memory.registered[mem]) {
            #define TEMPLATE(dtype) \
            if (view_mem.tensor_type == typeid(TensorView<dtype>)) { \
                TensorView<dtype>* view = reinterpret_cast<TensorView<dtype>*>(view_mem.address); \
                view->untie_base(); \
                continue; \
            }
            #include "specialize/all.h"

            log("tensor type is not recognized in memory, cannot free");
            abort();
        }
        memory.registered.erase(mem);
        memory.tensors.erase(std::find(memory.tensors.begin(), memory.tensors.end(), mem));
    }

    void Backend::mem_cacheinc(uint bytes) {
        memory.cache += bytes;
    }

    void Backend::mem_cacheclear() {
        memory.cache = 0;
    }

    uint Backend::mem_cacheget() const {
        return memory.cache;
    }


}