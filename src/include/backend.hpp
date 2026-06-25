#pragma once

#include <string>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <typeindex>
#include <vector>
#include <Metal/Metal.hpp>


namespace oxide {


    // pointers to gpu processes
    struct Metal {
        MTL::Device* device = nullptr; // pointer to gpu hardware device
        MTL::CommandQueue* queue = nullptr; // processing queue of instructions

        ~Metal();
    };


    // a collection of shader functions loaded from the source file
    struct Shader {
        MTL::Library* library = nullptr; // metal library object
        std::unordered_map<std::string, MTL::ComputePipelineState*> pipeline; // maps function name to MTL::ComputePipelineState object pointers

        ~Shader();
    };


    // backend randomization
    struct Random {
        std::random_device device;
        std::mt19937 generator;
    };


    // wrapper class for any type, used for memory management and registration
    struct TensorMemory {
        void* address = nullptr;
        std::type_index tensor_type;
        
        TensorMemory();
        TensorMemory(void* _address, std::type_index _tensor_type);

        bool operator==(const TensorMemory& other) const;
        bool valid() const;
    };


    // hasher for TensorMemory
    struct TensorMemoryHash {
        std::size_t operator()(const TensorMemory &x) const;
    };


    // auto memory management data and things
    struct Memory {
        std::unordered_map<TensorMemory, std::unordered_set<TensorMemory, TensorMemoryHash>, TensorMemoryHash> registered;
        std::vector<TensorMemory> tensors;
    };


    // bundled metadata, pointers, basic metal utilities
    // must be passed when communicating with metal framework
    class Backend {
        Metal metal;
        Shader shader;
        Random random;
        Memory memory;
        NS::Error* error = nullptr; // automatically points to error struct upon exception
        std::string error_log; // printed upon abort
        
        public:
            Backend();
            ~Backend();
            
            void init_metal();

            void init_shader(const std::string& path);
            void load_shader(const std::string& name); // load individual functions inside shader source file
            NS::UInteger set_cps(MTL::ComputeCommandEncoder* encoder, const std::string& name); // returns cps max threads

            MTL::Buffer* new_buffer(int size); // create new shared resource buffer
            MTL::CommandBuffer* new_cmd_buffer(); // create single-use command buffer

            NS::Error** error_out();
            void log(const std::string& message); // log error message
            void log_metal(); // automatically log metal error description if applicable
            void abort(); // throw runtime error

            std::mt19937& random_generate();

            TensorMemory memory_register(void* address, std::type_index tensor_type);
            TensorMemory memory_register(TensorMemory parent_memory, void* address, std::type_index tensor_type);
            void memory_unregister(TensorMemory parent_memory, TensorMemory view_memory);
            const std::vector<TensorMemory>& get_tensors() const;
            const std::unordered_set<TensorMemory, TensorMemoryHash>& memory_tied(TensorMemory key) const;
            void memory_delete(TensorMemory tensor_memory);
    };


}