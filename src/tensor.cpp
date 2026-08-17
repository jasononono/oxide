#include "tensor.hpp"
#include "common.hpp"


namespace oxide {

    
    template <typename dtype>
    TensorData<dtype>::TensorData(Backend& _backend, uint _size, dtype value):
        backend(&_backend), size(_size) {
        
        create_buffer();
        std::fill(ptr, ptr + size, value);

        mem = backend->mem_register(this, typeid(TensorData<dtype>));
    }

    template <typename dtype>
    TensorData<dtype>::~TensorData() {
        if (buffer) {buffer->release();}
    }

    template <typename dtype>
    TensorData<dtype>::TensorData(const TensorData<dtype>& other):
        size(other.size) {
        if (backend != other.get_backend()) {
            backend->log("backend mismatch");
            backend->abort();
        }
        
        create_buffer();
        std::memcpy(ptr, other.ptr, size * sizeof(dtype));

        mem = backend->mem_register(this, typeid(TensorData<dtype>));
    }

    template <typename dtype>
    TensorData<dtype>::TensorData(TensorData<dtype>&& other):
    backend(other.backend), buffer(other.buffer), ptr(other.ptr), size(other.size), mem(other.mem) {
        other.buffer = nullptr;
        other.ptr = nullptr;
        other.size = 0;
        other.mem = TensorMemory();
    }

    template <typename dtype>
    TensorData<dtype>& TensorData<dtype>::operator=(const TensorData<dtype>& other) {
        if (this == &other) {return *this;}
        if (backend != other.get_backend()) {
            backend->log("backend mismatch");
            backend->abort();
        }
        if (buffer) {buffer->release();}

        size = other.size;
        create_buffer();
        std::memcpy(ptr, other.ptr, size * sizeof(dtype));

        return *this;
    }

    template <typename dtype>
    TensorData<dtype>& TensorData<dtype>::operator=(TensorData<dtype>&& other) {
        if (this == &other) {return *this;}
        if (buffer) {buffer->release();}

        backend = other.backend;
        buffer = other.buffer;
        ptr = other.ptr;
        size = other.size;
        mem = other.mem;

        other.buffer = nullptr;
        other.ptr = nullptr;
        other.size = 0;
        other.mem = TensorMemory();

        return *this;
    }

    template <typename dtype>
    void TensorData<dtype>::create_buffer() {
        buffer = backend->new_buffer(size);
        ptr = static_cast<dtype*>(buffer->contents());
    }

    template <typename dtype>
    dtype TensorData<dtype>::operator[](iint index) const {
        check_buffer();
        return ptr[index];
    }

    template <typename dtype>
    dtype& TensorData<dtype>::operator[](iint index) {
        check_buffer();
        return ptr[index];
    }

    template <typename dtype>
    Backend* TensorData<dtype>::get_backend() const {
        return backend;
    }

    template <typename dtype>
    dtype* TensorData<dtype>::get_ptr() const {
        check_buffer();
        return ptr;
    }

    template <typename dtype>
    MTL::Buffer* TensorData<dtype>::get_buffer() const {
        check_buffer();
        return buffer;
    }

    template <typename dtype>
    uint TensorData<dtype>::get_size() const {
        return size;
    }

    template <typename dtype>
    std::string TensorData<dtype>::get_string() const {
        check_buffer();
        std::string str = "[";
        for (iint i = 0; i < size; i++) {
            str += std::to_string(ptr[i]);
            if (i == size - 1) {str += ']';}
            else {str += ", ";}
        }
        return str;
    }

    template <typename dtype>
    TensorMemory TensorData<dtype>::get_mem() const {
        return mem;
    }

    template <typename dtype>
    void TensorData<dtype>::check_buffer() const {
        if (!buffer) {
            backend->log("cannot access null buffer after move operation");
            backend->abort();
        }
    }

    #define TEMPLATE(dtype) template class TensorData<dtype>;
    #include "specialize/all.h"


    template <typename dtype>
    TensorView<dtype>::TensorView(Backend& _backend, const std::vector<uint>& _shape, TensorData<dtype>* _base):
    backend(&_backend), shape(_shape), base(_base), ndim(_shape.size()), strides(ndim) {
        if (backend != base->get_backend()) {
            backend->log("backend mismatch");
            backend->abort();
        }

        set_shape(_shape);

        if (base) {
            mem = backend->mem_register(base->get_mem(), this, typeid(TensorView<dtype>));
        }
    }

    template <typename dtype>
    TensorView<dtype>::TensorView(Backend& _backend, const std::vector<uint>& _shape, TensorData<dtype>* _base, iint _offset, const std::vector<iint>& _strides):
    backend(&_backend), shape(_shape), base(_base), ndim(_shape.size()), offset(_offset), strides(_strides) {
        if (backend != base->get_backend()) {
            backend->log("backend mismatch");
            backend->abort();
        }

        set_shape(_shape, _strides, _offset);

        if (base) {
            mem = backend->mem_register(base->get_mem(), this, typeid(TensorView<dtype>));
        }
    }

    template <typename dtype>
    TensorView<dtype>::~TensorView() {
        if (base && mem.valid()) {
            backend->mem_unregister(base->get_mem(), mem);
        }
    }

    template <typename dtype>
    TensorView<dtype>::TensorView(const TensorView<dtype>& other): backend(other.backend), base(other.base), ndim(other.ndim), size(other.size), offset(other.offset), shape(other.shape), strides(other.strides) {
        if (base) {
            mem = backend->mem_register(base->get_mem(), this, typeid(TensorView<dtype>));
        }
    }

    template <typename dtype>
    TensorView<dtype>::TensorView(TensorView<dtype>&& other):
    backend(other.backend), base(other.base), ndim(other.ndim), size(other.size), offset(other.offset), shape(other.shape), strides(other.strides), mem(other.mem) {
        other.base = nullptr;
        other.mem = TensorMemory();
    }

    template <typename dtype>
    TensorView<dtype>& TensorView<dtype>::operator=(const TensorView<dtype>& other) {
        if (this == &other) {return *this;}
        
        backend = other.backend;
        base = other.base;
        ndim = other.ndim;
        size = other.size;
        offset = other.offset;
        shape = other.shape;
        strides = other.strides;

        if (base) {
            mem = backend->mem_register(base->get_mem(), this, typeid(TensorView<dtype>));
        }

        return *this;
    }

    template <typename dtype>
    TensorView<dtype>& TensorView<dtype>::operator=(TensorView<dtype>&& other) {
        if (this == &other) {return *this;}

        backend = other.backend;
        base = other.base;
        ndim = other.ndim;
        size = other.size;
        offset = other.offset;
        shape = other.shape;
        strides = other.strides;
        mem = other.mem;

        other.base = nullptr;
        other.mem = TensorMemory();

        return *this;
    }

    template <typename dtype>
    dtype TensorView<dtype>::operator[](const std::vector<iint>& indices) const {
        return base->get_ptr()[get_buffer_idx(indices)];
    }

    template <typename dtype>
    dtype& TensorView<dtype>::operator[](const std::vector<iint>& indices) {
        return base->get_ptr()[get_buffer_idx(indices)];
    }

    template <typename dtype>
    dtype TensorView<dtype>::get_element(const std::vector<iint>& indices) {
        return (*this)[indices];
    }

    template <typename dtype>
    void TensorView<dtype>::set_element(const std::vector<iint>& indices, dtype value) {
        (*this)[indices] = value;
    }

    template <typename dtype>
    void TensorView<dtype>::set_shape(const std::vector<uint>& _shape) {
        shape = _shape;
        size = parse_shape(*backend, shape);
        ndim = shape.size();
        strides = std::vector<iint>(ndim);

        strides[ndim - 1] = 1;
        for (iint i = ndim - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    template <typename dtype>
    void TensorView<dtype>::set_shape(const std::vector<uint>& _shape, const std::vector<iint>& _strides, uint _offset) {
        shape = _shape;
        size = parse_shape(*backend, shape);
        ndim = shape.size();
        strides = _strides;
        offset = _offset;

        if (strides.size() != ndim) {
            backend->log("tensor strides must have the same dimensions as shape");
            backend->abort();
        }
    }

    template <typename dtype>
    iint TensorView<dtype>::get_buffer_idx(const std::vector<iint>& indices) const {
        check_base();
        if (indices.size() != ndim) {
            backend->log("indexing dimensions does not match tensor dimensions");
            backend->abort();
        }

        iint buf_index = 0, idx;
        for (iint i = 0; i < ndim; i++) {
            if (indices[i] >= 0) {
                idx = indices[i];
            } else {
                idx = shape[i] + indices[i];
            }
            if (idx < 0 || idx >= shape[i]) {
                backend->log("index out of range"); backend->abort();
            }
            buf_index += idx * strides[i];
        }

        return buf_index;
    }

    template <typename dtype>
    Backend* TensorView<dtype>::get_backend() const {
        return backend;
    }

    template <typename dtype>
    TensorData<dtype>* TensorView<dtype>::get_base() const {
        check_base();
        return base;
    }

    template <typename dtype>
    uint TensorView<dtype>::get_ndim() const {
        return ndim;
    }

    template <typename dtype>
    uint TensorView<dtype>::get_size() const {
        return size;
    }

    template <typename dtype>
    uint TensorView<dtype>::get_offset() const {
        return offset;
    }
    
    template <typename dtype>
    const std::vector<uint>& TensorView<dtype>::get_shape() const {
        return shape;
    }

    template <typename dtype>
    const std::vector<iint>& TensorView<dtype>::get_strides() const {
        return strides;
    }

    template <typename dtype>
    std::string TensorView<dtype>::get_string() const {
        std::string str(ndim, '[');
        std::vector<iint> indices(ndim, 0);
        
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

    template <typename dtype>
    void TensorView<dtype>::check_base() const {
        if (!base) {
            backend->log("base missing during tensor view operation");
            backend->abort();
        }
        if (backend != base->get_backend()) {
            backend->log("backend mismatch");
            backend->abort();
        }
    }

    #define TEMPLATE(dtype) template class TensorView<dtype>;
    #include "specialize/all.h"


    uint parse_shape(Backend& backend, const std::vector<uint>& shape) {
        if (shape.size() > MAXDIMS) {
            backend.log("tensor max dimensions exceeded");
            backend.abort();
        }
        if (shape.size() == 0) {
            backend.log("tensor shape must not be empty");
            backend.abort();
        }
        
        uint size = 1;
        for (uint i : shape) {
            if (i <= 0) {
                backend.log("dimension must be greater than 0");
                backend.abort();
            }
            size *= i;
        }
        return size;
    }


}