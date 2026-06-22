#pragma once

#include "backend.hpp"

#include <string>


namespace oxide {


    // collection of functions that launch gpu tasks
    // called by wrapper functions (which requires the dispatcher object)
    class Dispatcher {
        Backend* backend = nullptr;

        public:
            Dispatcher(Backend& _backend);

            Backend* get_backend() const;

            void binary_operand(const std::string& function, unsigned int size, MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* out, unsigned int ndim, const std::vector<int>& a_strides, unsigned int a_offset, const std::vector<int>& b_strides, unsigned int b_offset, const std::vector<int>& out_strides);
            void unary_operand(const std::string& function, unsigned int size, MTL::Buffer* a, MTL::Buffer* b, unsigned int ndim, const std::vector<int>& a_strides, unsigned int a_offset, const std::vector<int>& b_strides, unsigned int b_offset);
    };


}