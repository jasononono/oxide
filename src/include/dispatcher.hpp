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

            Backend* get_backend() const;

            void binary_operand(const std::string& function, uint size, MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* out, uint ndim, const std::vector<iint>& a_strides, uint a_offset, const std::vector<iint>& b_strides, uint b_offset, const std::vector<iint>& out_strides);
            void unary_operand(const std::string& function, uint size, MTL::Buffer* a, MTL::Buffer* b, uint ndim, const std::vector<iint>& a_strides, uint a_offset, const std::vector<iint>& b_strides, uint b_offset);
            
            void rand(uint size, MTL::Buffer* buf, uint seed); // [0, 1)
            template <typename d_type>
            void random(const std::string& function, uint size, MTL::Buffer* buf, uint seed, d_type a, d_type b); // [a, b]
            
    };


}