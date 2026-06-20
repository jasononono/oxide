#pragma once

#include <stack>
#include "backend.hpp"
#include "typeutil.hpp"


namespace oxide {


    // tensor class for storage only (contains only a linear size reference)
    template <typename d_type>
    class Tensor {
        Backend* backend = nullptr;
        d_type* ptr = nullptr;
        MTL::Buffer* buffer = nullptr;
        TensorMemory memory_reference;

        unsigned int size = 0;
        
        public:
            Tensor(Backend& _backend, unsigned int _size, d_type value);

            ~Tensor();
            Tensor(const Tensor& other);
            Tensor(Tensor&& other);
            Tensor& operator=(const Tensor& other);
            Tensor& operator=(Tensor&& other);

            void create_buffer(); // create buffer based on size attribute
            
            d_type operator[](int index) const;
            d_type& operator[](int index);

            Backend* get_backend() const;
            d_type* get_ptr() const;
            MTL::Buffer* get_buffer() const;
            unsigned int get_size() const;
            std::string get_string() const;
            TensorMemory get_memory_reference() const;

            void check_buffer() const; // throws error if buffer is null
    };


    template <typename d_type>
    class TensorView {
        Backend* backend = nullptr;
        Tensor<d_type>* base = nullptr;
        TensorMemory memory_reference;

        unsigned int ndim;
        unsigned int offset = 0;
        std::vector<unsigned int> shape;
        std::vector<int> strides;

        public:
            TensorView(Backend& _backend, const std::vector<unsigned int>& _shape, Tensor<d_type>* _base);
            TensorView(Backend& _backend, const std::vector<unsigned int>& _shape, Tensor<d_type>* _base, int _offset, const std::vector<int>& _strides);

            d_type operator[](const std::vector<int>& indices) const;
            d_type& operator[](const std::vector<int>& indices);

            int get_buffer_idx(const std::vector<int>& indices) const; // convert indices into buffer offset index

            Backend* get_backend() const;
            Tensor<d_type>* get_base() const;
            unsigned int get_ndim() const;
            unsigned int get_offset() const;
            const std::vector<unsigned int>& get_shape() const;
            const std::vector<int>& get_strides() const;
            std::string get_string() const;

            void check_base() const; // throws error if base is null
    };
    

    unsigned int parse_shape(Backend& backend, const std::vector<unsigned int>& shape); // throws error if shape is invalid, then returns accumulative size


}