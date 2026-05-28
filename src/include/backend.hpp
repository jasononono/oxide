#pragma once

#include <string>
#include <random>
#include <unordered_map>
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


    // bundled metadata, pointers, basic metal utilities
    // must be passed when communicating with metal framework
    class Backend {
        Metal metal;
        Shader shader;
        Random random;
        NS::Error* error = nullptr; // automatically points to error struct upon exception
        std::string error_log; // printed upon abort
        
        public:
            Backend();
            ~Backend();
            
            void init_metal();

            void init_shader(const std::string& path);
            void load_shader(const std::string& name); // load individual functions inside shader source file
            NS::UInteger set_cps(MTL::ComputeCommandEncoder* encoder, const std::string& name); // returns cps max threads

            void rand(float* start, int size); // fill a chunk of memory with random floats from 0-1

            MTL::Buffer* new_buffer(int size); // create new shared resource buffer
            MTL::CommandBuffer* new_cmd_buffer(); // create single-use command buffer

            NS::Error** error_out();
            void log(const std::string& message); // log error message
            void log_metal(); // automatically log metal error description if applicable
            void abort(); // throw runtime error
    };


}