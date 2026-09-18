/*
BACKEND.HPP

class backend: bundle of L1 shenanigans
    - stores metal API & handles
    - shader library, API for creating and using pipelines
    - random generator (generate seeds for shader)
    - tensor memory management, optimization
    - error handling/logging

struct TensorMemory: container that stores ptr info about any tensor type
*/


#pragma once

#include <Metal/Metal.hpp>
#include <string>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <typeindex>
#include <vector>
#include <iostream>
#include <fstream>
#include <limits>


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

        std::uniform_int_distribution<uint> seed_dist; // seeds are in the range [0, 2^32)
    };


    // wrapper class for any type, used for memory management and registration
    // use this only for tensor related classes
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
        std::unordered_map<TensorMemory, std::unordered_set<TensorMemory, TensorMemoryHash>, TensorMemoryHash> registered; // maps TensorData to TensorView
        std::vector<TensorMemory> tensors; // collection of all TensorData
        
        uint cache = 0; // when cache >= CACHETHRESHOLD, mem is optimized
    };


    // bundled metadata, pointers, basic metal utilities
    // must be passed when communicating with metal framework (higher-level functions can use Dispatcher directly)
    class Backend {
        Metal metal;
        Shader shader;
        Random random;
        Memory memory;
        
        NS::Error* mtl_err = nullptr; // automatically points to error struct upon exception
        std::string error_log; // printed upon abort()
        std::string warning_log; // printed upon flush()
        
        public:
            bool flush_warnings = false; // when set to true, use cout to display warnings as soon as they are logged (c++ interface only)

            Backend();
            ~Backend();
            
            void init_metal();

            void init_shader(const std::string& path);
            void load_shader(const std::string& name); // load individual functions inside shader source file
            NS::UInteger set_cps(MTL::ComputeCommandEncoder* encoder, const std::string& name); // returns cps max threads

            MTL::Buffer* new_buffer(uint size); // create new shared resource buffer
            MTL::CommandBuffer* new_cmd_buffer(); // create single-use command buffer

            NS::Error** get_mtl_err();
            void logw(const std::string& msg); // log warning message
            void log(const std::string& msg); // log error message
            void log_metal(); // automatically log metal error description if applicable
            void abort(); // throws error, Python system should translate the oxide_error thrown
            std::string flush(bool display = true); // flushes warnings, Python system should display warning_log after flushing

            std::mt19937& random_generate(); // returns the generator ONLY
            uint random_seed();

            TensorMemory mem_register(void* address, std::type_index tensor_type); // register TensorData
            TensorMemory mem_register(TensorMemory parent_mem, void* address, std::type_index tensor_type); // register TensorView
            void mem_unregister(TensorMemory parent_mem, TensorMemory view_mem); // unregister TensorView
            const std::vector<TensorMemory>& get_tensors() const;
            const std::unordered_set<TensorMemory, TensorMemoryHash>& get_mem_tied(TensorMemory key) const;
            void mem_delete(TensorMemory mem); // free TensorData

            void mem_cacheinc(uint bytes);
            void mem_cacheclear();
            uint mem_cacheget() const;
    };


}