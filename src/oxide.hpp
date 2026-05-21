#pragma once

#include <string>
#include <vector>
#include <random>
#include <unordered_map>
#include <Metal/Metal.hpp>


namespace ox {


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


    // tensor class for storage only (contains only a linear size reference)
    class Tensor {
        Backend* backend = nullptr;
        float* ptr = nullptr;
        MTL::Buffer* buffer = nullptr;
        int size = 0;
        
        public:
            Tensor(Backend& _backend, int _size, float value);

            ~Tensor();
            Tensor(const Tensor& other);
            Tensor(Tensor&& other);
            Tensor& operator=(const Tensor& other);
            Tensor& operator=(Tensor&& other);

            void create_buffer(); // create buffer based on size attribute
            
            float operator[](int index) const;
            float& operator[](int index);

            Backend* get_backend() const;
            float* get_ptr() const;
            MTL::Buffer* get_buffer() const;
            int get_size() const;
            std::string get_string() const;
    };


    // collection of functions that launch gpu tasks
    // called by wrapper functions (which requires the dispatcher object)
    class Dispatcher {
        Backend* backend = nullptr;

        public:
            Dispatcher(Backend& _backend);

            Backend* get_backend() const;

            void binary_operand(const std::string& function, int size, MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* out);
    };

    Tensor binary_add(Dispatcher& dispatcher, const Tensor& a, const Tensor& b); // elementwise local operation (e.g. a + b)


     Tensor rand(Backend& backend, int size); // generate tensor filled with random floats
}