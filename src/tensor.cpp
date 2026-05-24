#include "tensor.hpp"
#include "backend.hpp"

#include <string>
#include <algorithm>


namespace ox {

    
    template <typename d_type>
    Tensor<d_type>::Tensor(Backend& _backend, int _size, d_type value):
        backend(&_backend), size(_size) {

        create_buffer();
        std::fill(ptr, ptr + size, value);
    }

    template <typename d_type>
    Tensor<d_type>::~Tensor() {
        if (buffer) {buffer->release();}
    }

    template <typename d_type>
    Tensor<d_type>::Tensor(const Tensor<d_type>& other):
        backend(other.backend), size(other.size) {
        
        create_buffer();
        std::memcpy(ptr, other.ptr, size * sizeof(d_type));
    }

    template <typename d_type>
    Tensor<d_type>::Tensor(Tensor<d_type>&& other):
    backend(other.backend), buffer(other.buffer), ptr(other.ptr), size(other.size) {

        other.buffer = nullptr;
        other.ptr = nullptr;
        other.size = 0;
    }

    template <typename d_type>
    Tensor<d_type>& Tensor<d_type>::operator=(const Tensor<d_type>& other) {
        if (this == &other) {return *this;}
        if (buffer) {buffer->release();}

        backend = other.backend;
        size = other.size;
        create_buffer();
        std::memcpy(ptr, other.ptr, size * sizeof(d_type));

        return *this;
    }

    template <typename d_type>
    Tensor<d_type>& Tensor<d_type>::operator=(Tensor<d_type>&& other) {
        if (this == &other) {return *this;}
        if (buffer) {buffer->release();}

        backend = other.backend;
        buffer = other.buffer;
        ptr = other.ptr;
        size = other.size;

        other.buffer = nullptr;
        other.ptr = nullptr;
        other.size = 0;

        return *this;
    }

    template <typename d_type>
    void Tensor<d_type>::create_buffer() {
        buffer = backend->new_buffer(size);
        ptr = static_cast<d_type*>(buffer->contents());
    }

    template <typename d_type>
    d_type Tensor<d_type>::operator[](int index) const {
        if (!buffer) {
            backend->log("Oxide: cannot access null tensor after move operation."); backend->abort();
        }
        if (index < 0) {index += size;}
        if (index < 0 || index >= size) {
            backend->log("Oxide: tensor index out of range"); backend->abort();
        }
        return ptr[index];
    }

    template <typename d_type>
    d_type& Tensor<d_type>::operator[](int index) {
        if (!buffer) {
            backend->log("Oxide: cannot access null tensor after move operation."); backend->abort();
        }
        if (index < 0) {index += size;}
        if (index < 0 || index >= size) {
            backend->log("Oxide: tensor index out of range"); backend->abort();
        }
        return ptr[index];
    }

    template <typename d_type>
    Backend* Tensor<d_type>::get_backend() const {
        return backend;
    }

    template <typename d_type>
    d_type* Tensor<d_type>::get_ptr() const {
        return ptr;
    }

    template <typename d_type>
    MTL::Buffer* Tensor<d_type>::get_buffer() const {
        return buffer;
    }

    template <typename d_type>
    int Tensor<d_type>::get_size() const {
        return size;
    }

    template <typename d_type>
    std::string Tensor<d_type>::get_string() const {
        std::string str = "[";
        for (int i = 0; i < size; i++) {
            str += std::to_string(ptr[i]);
            if (i == size - 1) {str += ']';}
            else {str += ", ";}
        }
        return str;
    }


    template class Tensor<int32>;
    template class Tensor<float32>;


}