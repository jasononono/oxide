#pragma once

#include "backend.hpp"
#include "common.hpp"
#include <stack>
#include <string>
#include <algorithm>


namespace oxide {


    // tensor class for storage only (contains only a linear size reference)
    template <typename d_type>
    class TensorData {
        Backend* backend = nullptr;
        d_type* ptr = nullptr;
        MTL::Buffer* buffer = nullptr;
        TensorMemory mem = TensorMemory();

        uint size = 0;
        
        public:
            TensorData(Backend& _backend, uint _size, d_type value);

            ~TensorData();
            TensorData(const TensorData& other);
            TensorData(TensorData&& other);
            TensorData& operator=(const TensorData& other);
            TensorData& operator=(TensorData&& other);

            void create_buffer(); // create buffer based on size attribute
            
            d_type operator[](iint index) const;
            d_type& operator[](iint index);

            Backend* get_backend() const;
            d_type* get_ptr() const;
            MTL::Buffer* get_buffer() const;
            uint get_size() const;
            std::string get_string() const;
            TensorMemory get_mem() const;

            void check_buffer() const; // throws error if buffer is null
    };


    template <typename d_type>
    class TensorView {
        Backend* backend = nullptr;
        TensorData<d_type>* base = nullptr;
        TensorMemory mem = TensorMemory();

        uint ndim, size;
        uint offset = 0;
        std::vector<uint> shape;
        std::vector<iint> strides;

        public:
            TensorView(Backend& _backend, const std::vector<uint>& _shape, TensorData<d_type>* _base);
            TensorView(Backend& _backend, const std::vector<uint>& _shape, TensorData<d_type>* _base, iint _offset, const std::vector<iint>& _strides);

            ~TensorView();
            TensorView(const TensorView& other);
            TensorView(TensorView&& other);
            TensorView& operator=(const TensorView& other);
            TensorView& operator=(TensorView&& other);

            d_type operator[](const std::vector<iint>& indices) const;
            d_type& operator[](const std::vector<iint>& indices);
            
            d_type get_element(const std::vector<iint>& indices);
            void set_element(const std::vector<iint>& indices, d_type value);

            iint get_buffer_idx(const std::vector<iint>& indices) const; // convert indices into buffer offset index

            void set_shape(const std::vector<uint>& _shape);
            void set_shape(const std::vector<uint>& _shape, const std::vector<iint>& _strides, uint _offset);

            Backend* get_backend() const;
            TensorData<d_type>* get_base() const;
            uint get_ndim() const;
            uint get_size() const;
            uint get_offset() const;
            const std::vector<uint>& get_shape() const;
            const std::vector<iint>& get_strides() const;
            std::string get_string() const;

            void check_base() const; // throws error if base is null
    };
    

    uint parse_shape(Backend& backend, const std::vector<uint>& shape); // throws error if shape is invalid, then returns accumulative size


}