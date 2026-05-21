#include "oxide.hpp"

#include <string>
#include <algorithm>


namespace ox {


    Tensor::Tensor(Backend& _backend, int _size, float value):
        backend(&_backend), size(_size) {

        create_buffer();
        std::fill(ptr, ptr + size, value);
    }

    Tensor::~Tensor() {
        if (buffer) {buffer->release();}
    }

    Tensor::Tensor(const Tensor& other):
        backend(other.backend), size(other.size) {
        
        create_buffer();
        std::memcpy(ptr, other.ptr, size * sizeof(float));
    }

    Tensor::Tensor(Tensor&& other):
    backend(other.backend), buffer(other.buffer), ptr(other.ptr), size(other.size) {

        other.buffer = nullptr;
        other.ptr = nullptr;
        other.size = 0;
    }

    Tensor& Tensor::operator=(const Tensor& other) {
        if (this == &other) {return *this;}
        if (buffer) {buffer->release();}

        backend = other.backend;
        size = other.size;
        create_buffer();
        std::memcpy(ptr, other.ptr, size * sizeof(float));

        return *this;
    }

    Tensor& Tensor::operator=(Tensor&& other) {
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

    void Tensor::create_buffer() {
        buffer = backend->new_buffer(size);
        ptr = static_cast<float*>(buffer->contents());
    }

    float Tensor::operator[](int index) const {
        if (!buffer) {
            backend->log("Oxide: cannot access null tensor after move operation."); backend->abort();
        }
        if (index < 0) {index += size;}
        if (index < 0 || index >= size) {
            backend->log("Oxide: tensor index out of range"); backend->abort();
        }
        return ptr[index];
    }

    float& Tensor::operator[](int index) {
        if (!buffer) {
            backend->log("Oxide: cannot access null tensor after move operation."); backend->abort();
        }
        if (index < 0) {index += size;}
        if (index < 0 || index >= size) {
            backend->log("Oxide: tensor index out of range"); backend->abort();
        }
        return ptr[index];
    }

    Backend* Tensor::get_backend() const {
        return backend;
    }

    float* Tensor::get_ptr() const {
        return ptr;
    }

    MTL::Buffer* Tensor::get_buffer() const {
        return buffer;
    }

    int Tensor::get_size() const {
        return size;
    }

    std::string Tensor::get_string() const {
        std::string str = "[";
        for (int i = 0; i < size; i++) {
            str += std::to_string(ptr[i]);
            if (i == size - 1) {str += ']';}
            else {str += ", ";}
        }
        return str;
    }


    Tensor rand(Backend& backend, int size) {
        Tensor out(backend, size, 0);
        backend.rand(out.get_ptr(), size);
        return out;
    }


}