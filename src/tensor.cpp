#include "tensor.hpp"
#include "backend.hpp"

#include <string>
#include <algorithm>
#include <iostream>


namespace oxide {

    
    template <typename d_type>
    TensorData<d_type>::TensorData(Backend& _backend, unsigned int _size, d_type value):
        backend(&_backend), size(_size) {
        
        create_buffer();
        std::fill(ptr, ptr + size, value);

        memory_reference = backend->memory_register(this, typeid(TensorData<d_type>));
    }

    template <typename d_type>
    TensorData<d_type>::~TensorData() {
        if (buffer) {buffer->release();}
    }

    template <typename d_type>
    TensorData<d_type>::TensorData(const TensorData<d_type>& other):
        size(other.size) {
        if (backend != other.get_backend()) {
            backend->log("Oxide: backend mismatch");
            backend->abort();
        }
        
        create_buffer();
        std::memcpy(ptr, other.ptr, size * sizeof(d_type));

        memory_reference = backend->memory_register(this, typeid(TensorData<d_type>));
    }

    template <typename d_type>
    TensorData<d_type>::TensorData(TensorData<d_type>&& other):
    backend(other.backend), buffer(other.buffer), ptr(other.ptr), size(other.size), memory_reference(other.memory_reference) {
        other.buffer = nullptr;
        other.ptr = nullptr;
        other.size = 0;
        other.memory_reference = TensorMemory();
    }

    template <typename d_type>
    TensorData<d_type>& TensorData<d_type>::operator=(const TensorData<d_type>& other) {
        if (this == &other) {return *this;}
        if (backend != other.get_backend()) {
            backend->log("Oxide: backend mismatch");
            backend->abort();
        }
        if (buffer) {buffer->release();}

        size = other.size;
        create_buffer();
        std::memcpy(ptr, other.ptr, size * sizeof(d_type));

        return *this;
    }

    template <typename d_type>
    TensorData<d_type>& TensorData<d_type>::operator=(TensorData<d_type>&& other) {
        if (this == &other) {return *this;}
        if (buffer) {buffer->release();}

        backend = other.backend;
        buffer = other.buffer;
        ptr = other.ptr;
        size = other.size;
        memory_reference = other.memory_reference;

        other.buffer = nullptr;
        other.ptr = nullptr;
        other.size = 0;
        other.memory_reference = TensorMemory();

        return *this;
    }

    template <typename d_type>
    void TensorData<d_type>::create_buffer() {
        buffer = backend->new_buffer(size);
        ptr = static_cast<d_type*>(buffer->contents());
    }

    template <typename d_type>
    d_type TensorData<d_type>::operator[](int index) const {
        check_buffer();
        return ptr[index];
    }

    template <typename d_type>
    d_type& TensorData<d_type>::operator[](int index) {
        check_buffer();
        return ptr[index];
    }

    template <typename d_type>
    Backend* TensorData<d_type>::get_backend() const {
        return backend;
    }

    template <typename d_type>
    d_type* TensorData<d_type>::get_ptr() const {
        check_buffer();
        return ptr;
    }

    template <typename d_type>
    MTL::Buffer* TensorData<d_type>::get_buffer() const {
        check_buffer();
        return buffer;
    }

    template <typename d_type>
    unsigned int TensorData<d_type>::get_size() const {
        return size;
    }

    template <typename d_type>
    std::string TensorData<d_type>::get_string() const {
        check_buffer();
        std::string str = "[";
        for (int i = 0; i < size; i++) {
            str += std::to_string(ptr[i]);
            if (i == size - 1) {str += ']';}
            else {str += ", ";}
        }
        return str;
    }

    template <typename d_type>
    TensorMemory TensorData<d_type>::get_memory_reference() const {
        return memory_reference;
    }

    template <typename d_type>
    void TensorData<d_type>::check_buffer() const {
        if (!buffer) {
            backend->log("Oxide: cannot access null buffer after move operation"); backend->abort();
        }
    }

    template class TensorData<int32>;
    template class TensorData<float32>;


    template <typename d_type>
    TensorView<d_type>::TensorView(Backend& _backend, const std::vector<unsigned int>& _shape, TensorData<d_type>* _base):
    backend(&_backend), shape(_shape), base(_base), ndim(_shape.size()), strides(ndim) {
        if (backend != base->get_backend()) {
            backend->log("Oxide: backend mismatch");
            backend->abort();
        }

        set_shape(_shape);

        if (base) {
            memory_reference = backend->memory_register(base->get_memory_reference(), this, typeid(TensorView<d_type>));
        }
    }

    template <typename d_type>
    TensorView<d_type>::TensorView(Backend& _backend, const std::vector<unsigned int>& _shape, TensorData<d_type>* _base, int _offset, const std::vector<int>& _strides):
    backend(&_backend), shape(_shape), base(_base), ndim(_shape.size()), offset(_offset), strides(_strides) {
        if (backend != base->get_backend()) {
            backend->log("Oxide: backend mismatch");
            backend->abort();
        }

        set_shape(_shape, _strides, _offset);

        if (base) {
            memory_reference = backend->memory_register(base->get_memory_reference(), this, typeid(TensorView<d_type>));
        }
    }

    template <typename d_type>
    TensorView<d_type>::~TensorView() {
        if (base && memory_reference.valid()) {
            backend->memory_unregister(base->get_memory_reference(), memory_reference);
        }
    }

    template <typename d_type>
    TensorView<d_type>::TensorView(const TensorView<d_type>& other):
    base(other.base), ndim(other.ndim), size(other.size), offset(other.offset), shape(other.shape), strides(other.strides) {
        if (backend != other.get_backend()) {
            backend->log("Oxide: backend mismatch");
            backend->abort();
        }
        if (base) {
            memory_reference = backend->memory_register(base->get_memory_reference(), this, typeid(TensorView<d_type>));
        }
    }

    template <typename d_type>
    TensorView<d_type>::TensorView(TensorView<d_type>&& other):
    backend(other.backend), base(other.base), ndim(other.ndim), size(other.size), offset(other.offset), shape(other.shape), strides(other.strides), memory_reference(other.memory_reference) {
        other.base = nullptr;
        other.memory_reference = TensorMemory();
    }

    template <typename d_type>
    TensorView<d_type>& TensorView<d_type>::operator=(const TensorView<d_type>& other) {
        if (this == &other) {return *this;}
        if (backend != other.get_backend()) {
            backend->log("Oxide: backend mismatch");
            backend->abort();
        }

        base = other.base;
        ndim = other.ndim;
        size = other.size;
        offset = other.offset;
        shape = other.shape;
        strides = other.strides;

        return *this;
    }

    template <typename d_type>
    TensorView<d_type>& TensorView<d_type>::operator=(TensorView<d_type>&& other) {
        if (this == &other) {return *this;}

        backend = other.backend;
        base = other.base;
        ndim = other.ndim;
        size = other.size;
        offset = other.offset;
        shape = other.shape;
        strides = other.strides;
        memory_reference = other.memory_reference;

        other.base = nullptr;
        other.memory_reference = TensorMemory();

        return *this;
    }

    template <typename d_type>
    d_type TensorView<d_type>::operator[](const std::vector<int>& indices) const {
        return base->get_ptr()[get_buffer_idx(indices)];
    }

    template <typename d_type>
    d_type& TensorView<d_type>::operator[](const std::vector<int>& indices) {
        return base->get_ptr()[get_buffer_idx(indices)];
    }

    template <typename d_type>
    void TensorView<d_type>::set_shape(const std::vector<unsigned int>& _shape) {
        shape = _shape;
        size = parse_shape(*backend, shape);
        ndim = shape.size();
        strides = std::vector<int>(ndim);

        strides[ndim - 1] = 1;
        for (int i = ndim - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    template <typename d_type>
    void TensorView<d_type>::set_shape(const std::vector<unsigned int>& _shape, const std::vector<int>& _strides, unsigned int _offset) {
        shape = _shape;
        size = parse_shape(*backend, shape);
        ndim = shape.size();
        strides = _strides;
        offset = _offset;

        if (strides.size() != ndim) {
            backend->log("Oxide: tensor strides must have the same dimensions as shape");
            backend->abort();
        }
    }

    template <typename d_type>
    int TensorView<d_type>::get_buffer_idx(const std::vector<int>& indices) const {
        check_base();
        if (indices.size() != ndim) {
            backend->log("Oxide: indexing dimensions does not match tensor dimensions"); backend->abort();
        }

        int buf_index = 0, idx;
        for (int i = 0; i < ndim; i++) {
            if (indices[i] >= 0) {
                idx = indices[i];
            } else {
                idx = shape[i] + indices[i];
            }
            if (idx < 0 || idx >= shape[i]) {
                backend->log("Oxide: index out of range"); backend->abort();
            }
            buf_index += idx * strides[i];
        }

        return buf_index;
    }

    template <typename d_type>
    Backend* TensorView<d_type>::get_backend() const {
        return backend;
    }

    template <typename d_type>
    TensorData<d_type>* TensorView<d_type>::get_base() const {
        check_base();
        return base;
    }

    template <typename d_type>
    unsigned int TensorView<d_type>::get_ndim() const {
        return ndim;
    }

    template <typename d_type>
    unsigned int TensorView<d_type>::get_size() const {
        return size;
    }

    template <typename d_type>
    unsigned int TensorView<d_type>::get_offset() const {
        return offset;
    }
    
    template <typename d_type>
    const std::vector<unsigned int>& TensorView<d_type>::get_shape() const {
        return shape;
    }

    template <typename d_type>
    const std::vector<int>& TensorView<d_type>::get_strides() const {
        return strides;
    }

    template <typename d_type>
    std::string TensorView<d_type>::get_string() const {
        std::string str(ndim, '[');
        std::vector<int> indices(ndim, 0);
        
        while (!indices.empty()) {
            str += std::to_string((*this)[indices]);
            while (!indices.empty() && indices.back() == shape[indices.size() - 1] - 1) {
                str += ']';
                indices.pop_back();
            }
            if (indices.empty()) {break;}
            str += ", ";
            indices.back()++;
            while (indices.size() < ndim) {
                str += '[';
                indices.push_back(0);
            }
        }
        return str;
    }

    template <typename d_type>
    void TensorView<d_type>::check_base() const {
        if (!base) {
            backend->log("Oxide: base missing during tensor view operation"); backend->abort();
        }
        if (backend != base->get_backend()) {
            backend->log("Oxide: backend mismatch");
            backend->abort();
        }
    }

    template class TensorView<int32>;
    template class TensorView<float32>;


    unsigned int parse_shape(Backend& backend, const std::vector<unsigned int>& shape) {
        if (shape.size() > MAXDIMS) {
            backend.log("Oxide: tensor max dimensions exceeded");
            backend.abort();
        }
        if (shape.size() == 0) {
            backend.log("Oxide: tensor shape must not be empty");
            backend.abort();
        }
        
        unsigned int size = 1;
        for (unsigned int i : shape) {
            if (i <= 0) {
                backend.log("Oxide: dimension must be greater than 0"); backend.abort();
            }
            size *= i;
        }
        return size;
    }


}