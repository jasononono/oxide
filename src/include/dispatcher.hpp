/*
DISPATCHER.HPP

class Dispatcher: L2 dispatcher implementation, configures and begins GPU processes
*/


#pragma once

#include "backend.hpp"
#include "common.hpp"
#include <string>


namespace oxide {


    // collection of functions that launch gpu tasks
    // called by wrapper functions (which requires the dispatcher object)
    class Dispatcher {
        Backend* backend = nullptr;

        public:
            Dispatcher(Backend& _backend);
            ~Dispatcher();

            Backend* get_backend() const;

            void binary_operation(const std::string& function, uint size, MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* out, uint ndim, const std::vector<iint>& a_strides, uint a_offset, const std::vector<iint>& b_strides, uint b_offset, const std::vector<iint>& out_strides); // e.g. out = a + b
            void unary_operation(const std::string& function, uint size, MTL::Buffer* a, MTL::Buffer* b, uint ndim, const std::vector<iint>& a_strides, uint a_offset, const std::vector<iint>& b_strides, uint b_offset); // e.g. a += b
            
            void rand(uint size, MTL::Buffer* buf, uint seed); // random float32 in [0, 1)
            template <typename dtype>
            void random(const std::string& function, uint size, MTL::Buffer* buf, uint seed, dtype a, dtype b); // random numeric in [a, b]
            
            void memcpy(uint size, MTL::Buffer* a, MTL::Buffer* b, uint ndim, const std::vector<iint>& a_strides, uint a_offset, const std::vector<iint>& b_strides, uint b_offset);
    };


}