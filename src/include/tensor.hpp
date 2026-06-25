#pragma once

#include <stack>
#include "backend.hpp"
#include "typeutil.hpp"


namespace oxide {


    // tensor class for storage only (contains only a linear size reference)
    template <typename d_type>
    class TensorData {
        Backend* backend = nullptr;
        d_type* ptr = nullptr;
        MTL::Buffer* buffer = nullptr;
        TensorMemory memory_reference = TensorMemory();

        uint size = 0;
        
        public:
            TensorData(Backend& _backend, uint _size, d_type value);

            ~TensorData();
            TensorData(const TensorData& other);
            TensorData(TensorData&& other);
            TensorData& operator=(const TensorData& other);
            TensorData& operator=(TensorData&& other);

            void create_buffer(); // create buffer based on size attribute
            
            d_type operator[](int index) const;
            d_type& operator[](int index);

            Backend* get_backend() const;
            d_type* get_ptr() const;
            MTL::Buffer* get_buffer() const;
            uint get_size() const;
            std::string get_string() const;
            TensorMemory get_memory_reference() const;

            void check_buffer() const; // throws error if buffer is null
    };


    template <typename d_type>
    class TensorView {
        Backend* backend = nullptr;
        TensorData<d_type>* base = nullptr;
        TensorMemory memory_reference = TensorMemory();

        uint ndim, size;
        uint offset = 0;
        std::vector<uint> shape;
        std::vector<int> strides;

        public:
            TensorView(Backend& _backend, const std::vector<uint>& _shape, TensorData<d_type>* _base);
            TensorView(Backend& _backend, const std::vector<uint>& _shape, TensorData<d_type>* _base, int _offset, const std::vector<int>& _strides);

            ~TensorView();
            TensorView(const TensorView& other);
            TensorView(TensorView&& other);
            TensorView& operator=(const TensorView& other);
            TensorView& operator=(TensorView&& other);

            d_type operator[](const std::vector<int>& indices) const;
            d_type& operator[](const std::vector<int>& indices);
            
            d_type get_element(const std::vector<int>& indices);
            void set_element(const std::vector<int>& indices, d_type value);

            int get_buffer_idx(const std::vector<int>& indices) const; // convert indices into buffer offset index

            void set_shape(const std::vector<uint>& _shape);
            void set_shape(const std::vector<uint>& _shape, const std::vector<int>& _strides, uint _offset);

            Backend* get_backend() const;
            TensorData<d_type>* get_base() const;
            uint get_ndim() const;
            uint get_size() const;
            uint get_offset() const;
            const std::vector<uint>& get_shape() const;
            const std::vector<int>& get_strides() const;
            std::string get_string() const;

            void check_base() const; // throws error if base is null
    };
    

    uint parse_shape(Backend& backend, const std::vector<uint>& shape); // throws error if shape is invalid, then returns accumulative size


}