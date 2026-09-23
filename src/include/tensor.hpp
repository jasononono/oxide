/*
TENSOR.HPP

class TensorData: object that owns the actual data of tensors
    - associated with a backend, supports basic indexing
    - contains a ptr to the data, freed automatically in mem checks implemented in oxide.hpp & backend.hpp
    - always call check_buffer() before using the underlying buffer (the object might be moved or uninitialized)

class TensorView: object tied to TensorData with shape, strides, and offset. In Python, it is stored as tensorclass.Tensor.ctensor
    - always call check_base() before using the tied TensorData (the object might be moved or uninitialized)
*/


#pragma once

#include "backend.hpp"
#include "common.hpp"
#include <stack>
#include <string>
#include <algorithm>


namespace oxide {


    // tensor class for storage only (contains only a linear size reference)
    template <typename dtype>
    class TensorData {
        Backend* backend = nullptr;
        dtype* ptr = nullptr;
        MTL::Buffer* buffer = nullptr;
        TensorMemory mem = TensorMemory();

        uint size = 0;
        
        public:
            TensorData(Backend& _backend, uint _size, dtype value);

            // rule of five
            ~TensorData();
            TensorData(const TensorData& other);
            TensorData(TensorData&& other);
            TensorData& operator=(const TensorData& other);
            TensorData& operator=(TensorData&& other);

            void create_buffer(); // create buffer based on size attribute
            
            // indexing functions
            const dtype& operator[](iint index) const;
            dtype& operator[](iint index);

            // getter functions
            Backend* get_backend() const;
            dtype* get_ptr() const;
            MTL::Buffer* get_buffer() const;
            uint get_size() const;
            std::string get_string() const;
            TensorMemory get_mem() const;

            void check_buffer() const; // throws error if buffer is null
    };


    template <typename dtype>
    class TensorView {
        Backend* backend = nullptr;
        TensorData<dtype>* base = nullptr;
        TensorMemory mem = TensorMemory();

        uint ndim, size;
        uint offset = 0;
        std::vector<uint> shape;
        std::vector<iint> strides;

        public:
            TensorView(Backend& _backend, const std::vector<uint>& _shape, TensorData<dtype>* _base);
            TensorView(Backend& _backend, const std::vector<uint>& _shape, TensorData<dtype>* _base, iint _offset);
            TensorView(Backend& _backend, const std::vector<uint>& _shape, TensorData<dtype>* _base, iint _offset, const std::vector<iint>& _strides);

            // rule of five
            ~TensorView();
            TensorView(const TensorView& other);
            TensorView(TensorView&& other);
            TensorView& operator=(const TensorView& other) &; // copy constructor, called when view is an lvalue (e.g. view = other;)
            TensorView& operator=(TensorView&& other) &; // move constructor, called when view is an lvalue (e.g. view = other;)

            // indexing
            const dtype& get_element(const std::vector<iint>& indices) const; // get a singular element (length of indices must match ndim)
            void set_element(const std::vector<iint>& indices, dtype value); // set a singular element (length of indices must match ndim)

            TensorView operator[](const std::vector<iint>& indices) const; // get a subarray or element as a TensorView
            TensorView& operator=(const TensorView& other) const&&; // copy data only, called when view is an rvalue (e.g. view[{}] = other;)

            iint get_buffer_idx(const std::vector<iint>& indices) const; // convert indices into buffer offset index

            // 0d views
            bool constant() const; // returns true if the shape is []
            const dtype& value() const;

            // reshape with size checking
            void set_shape(const std::vector<uint>& _shape); // this function will recalculate all strides (won't generate the correct reshaped view if the tensor is non-contiguous)
            void set_shape(const std::vector<uint>& _shape, const std::vector<iint>& _strides, uint _offset);
            bool contiguous() const; // returns true if the view occupies a contiguous chunk of buffer memory

            // getter functions
            Backend* get_backend() const;
            TensorData<dtype>* get_base() const;
            uint get_ndim() const;
            uint get_size() const;
            uint get_offset() const;
            const std::vector<uint>& get_shape() const;
            const std::vector<iint>& get_strides() const;
            std::string get_string() const;

            void check_base() const; // throws error if base is null
            void untie_base(); // set base to nullptr
    };
    

}