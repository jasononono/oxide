#pragma once

#include "backend.hpp"
#include "typeutil.hpp"


namespace ox {


    // tensor class for storage only (contains only a linear size reference)
    template <typename d_type>
    class Tensor {
        Backend* backend = nullptr;
        d_type* ptr = nullptr;
        MTL::Buffer* buffer = nullptr;
        int size = 0;
        
        public:
            Tensor(Backend& _backend, int _size, d_type value);

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
            int get_size() const;
            std::string get_string() const;
    };


}