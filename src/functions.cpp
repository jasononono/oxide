#include "functions.hpp"


namespace oxide {


    template <typename dtype>
    uint broadcast(Dispatcher& dispatcher, const TensorView<dtype>& a, const TensorView<dtype>& b, const uint ndim, std::vector<uint>& out_shape, std::vector<iint>& a_strides, std::vector<iint>& b_strides) {
        if (a.get_backend() != b.get_backend() || a.get_backend() != dispatcher.get_backend()) {
            dispatcher.get_backend()->log("backend mismatch");
            dispatcher.get_backend()->abort();
        }
        a.check_base(); b.check_base();

        uint size = 1;
        uint idx, a_idx, b_idx;

        for (uint i = 0; i < ndim; i++) {
            idx = ndim - i - 1;
            a_idx = a.get_ndim() - i - 1;
            b_idx = b.get_ndim() - i - 1;

            if (i >= a.get_ndim()) {
                out_shape[idx] = b.get_shape()[b_idx];
                a_strides[idx] = 0;
                b_strides[idx] = b.get_strides()[b_idx];
            } else if (i >= b.get_ndim()) {
                out_shape[idx] = a.get_shape()[a_idx];
                a_strides[idx] = a.get_strides()[a_idx];
                b_strides[idx] = 0;
            } else if (a.get_shape()[a_idx] == 1) {
                out_shape[idx] = b.get_shape()[b_idx];
                a_strides[idx] = 0;
                b_strides[idx] = b.get_strides()[b_idx];
            } else if (b.get_shape()[b_idx] == 1) {
                out_shape[idx] = a.get_shape()[a_idx];
                a_strides[idx] = a.get_strides()[a_idx];
                b_strides[idx] = 0;
            } else if (a.get_shape()[a_idx] == b.get_shape()[b_idx]) {
                out_shape[idx] = a.get_shape()[a_idx];
                a_strides[idx] = a.get_strides()[a_idx];
                b_strides[idx] = b.get_strides()[b_idx];
            } else {
                dispatcher.get_backend()->log("tensors cannot be broadcasted");
                dispatcher.get_backend()->abort();
            }

            size *= out_shape[idx];
        }

        return size;
    }
    #define TEMPLATE(dtype) template uint broadcast(Dispatcher& dispatcher, const TensorView<dtype>& a, const TensorView<dtype>& b, const uint ndim, std::vector<uint>& out_shape, std::vector<iint>& a_strides, std::vector<iint>& b_strides);
    #include "specialize/all.h"

    template <typename dtype>
    void ubroadcast(Dispatcher& dispatcher, const TensorView<dtype>& a, const TensorView<dtype>& b, std::vector<iint>& b_strides) {
        if (a.get_backend() != b.get_backend() || a.get_backend() != dispatcher.get_backend()) {
            dispatcher.get_backend()->log("backend mismatch");
            dispatcher.get_backend()->abort();
        }
        a.check_base(); b.check_base();
        
        if (b.get_ndim() > a.get_ndim()) {
            dispatcher.get_backend()->log("tensors cannot be broadcasted");
            dispatcher.get_backend()->abort();
        }

        uint idx, b_idx;

        for (iint i = 0; i < a.get_ndim(); i++) {
            idx = a.get_ndim() - i - 1;
            b_idx = b.get_ndim() - i - 1;

            if (i >= b.get_ndim()) {
                b_strides[idx] = 0;
            } else if (b.get_shape()[b_idx] == 1) {
                b_strides[idx] = 0;
            } else if (a.get_shape()[idx] == b.get_shape()[b_idx]) {
                b_strides[idx] = b.get_strides()[b_idx];
            } else {
                dispatcher.get_backend()->log("tensors cannot be broadcasted");
                dispatcher.get_backend()->abort();
            }
        }
    }
    #define TEMPLATE(dtype) void ubroadcast(Dispatcher& dispatcher, const TensorView<dtype>& a, const TensorView<dtype>& b, std::vector<iint>& b_strides);
    #include "specialize/all.h"


    #define SPEC2D
    #define TEMPLATE2D(dtype) \
    TEMPLATE(add, dtype) \
    TEMPLATE(sub, dtype) \
    TEMPLATE(mul, dtype) \
    TEMPLATE(div, dtype)

    #define TEMPLATE(op, null) template <typename dtype> \
    TensorView<dtype> op(Dispatcher& dispatcher, const TensorView<dtype>& a, const TensorView<dtype>& b) { \
        uint ndim = std::max(a.get_ndim(), b.get_ndim()); \
        std::vector<uint> out_shape(ndim); \
        std::vector<iint> a_strides(ndim), b_strides(ndim); \
        uint size = broadcast<dtype>(dispatcher, a, b, ndim, out_shape, a_strides, b_strides); \
        \
        TensorData<dtype>* out = new TensorData<dtype>(*dispatcher.get_backend(), size, 0); \
        TensorView<dtype> view(*dispatcher.get_backend(), out_shape, out); \
        \
        dispatcher.binary_operand(with_type<dtype>(#op), view.get_size(), a.get_base()->get_buffer(), b.get_base()->get_buffer(), out->get_buffer(), ndim, a_strides, a.get_offset(), b_strides, b.get_offset(), view.get_strides()); \
        return view; \
    }
    TEMPLATE2D(null)
    #undef TEMPLATE
    #define TEMPLATE(op, dtype) template TensorView<dtype> op(Dispatcher& dispatcher, const TensorView<dtype>& a, const TensorView<dtype>& b);
    #include "specialize/numeric.h"
    
    #define TEMPLATE(op, null) template <typename dtype> \
    TensorView<dtype>& u##op(Dispatcher& dispatcher, TensorView<dtype>& a, const TensorView<dtype>& b) { \
        uint ndim = a.get_ndim(); \
        std::vector<iint> b_strides(ndim); \
        uint size = a.get_size(); \
        ubroadcast<dtype>(dispatcher, a, b, b_strides); \
        \
        dispatcher.unary_operand(with_type<dtype>("u" #op), size, a.get_base()->get_buffer(), b.get_base()->get_buffer(), ndim, a.get_strides(), a.get_offset(), b_strides, b.get_offset()); \
        return a; \
    }
    TEMPLATE2D(null)
    #undef TEMPLATE
    #define TEMPLATE(op, dtype) template TensorView<dtype>& u##op(Dispatcher& dispatcher, TensorView<dtype>& a, const TensorView<dtype>& b);
    #include "specialize/numeric.h"

    #undef TEMPLATE
    #undef SPEC2D


    TensorView<float32> rand(Dispatcher& dispatcher, const std::vector<uint>& shape) {
        uint size = parse_shape(*dispatcher.get_backend(), shape);
        TensorData<float32>* out = new TensorData<float32>(*dispatcher.get_backend(), size, float32());

        dispatcher.rand(size, out->get_buffer(), dispatcher.get_backend()->random_seed());

        return TensorView<float32>(*dispatcher.get_backend(), shape, out);
    }

    template <typename dtype>
    TensorView<dtype> random(Dispatcher& dispatcher, const std::vector<uint>& shape, dtype a, dtype b) {
        uint size = parse_shape(*dispatcher.get_backend(), shape);
        TensorData<dtype>* out = new TensorData<dtype>(*dispatcher.get_backend(), size, dtype());

        dispatcher.random(with_type<dtype>("random"), size, out->get_buffer(), dispatcher.get_backend()->random_seed(), a, b);

        return TensorView<dtype>(*dispatcher.get_backend(), shape, out);
    }
    #define TEMPLATE(dtype) template TensorView<dtype> random(Dispatcher& dispatcher, const std::vector<uint>& shape, dtype a, dtype b);
    #include "specialize/numeric.h"


    template <typename dtype>
    TensorView<dtype> filled(Backend& backend, const std::vector<uint>& shape, dtype value) {
        uint size = parse_shape(backend, shape);
        TensorData<dtype>* out = new TensorData<dtype>(backend, size, value);
        return TensorView<dtype>(backend, shape, out);
    }
    #define TEMPLATE(dtype) template TensorView<dtype> filled(Backend& backend, const std::vector<uint>& shape, dtype value);
    #include "specialize/all.h"

    template <typename dtype>
    TensorView<dtype> zeros(Backend& backend, const std::vector<uint>& shape) {
        return filled<dtype>(backend, shape, 0);
    }
    #define TEMPLATE(dtype) template TensorView<dtype> zeros(Backend& backend, const std::vector<uint>& shape);
    #include "specialize/numeric.h"

    template <typename dtype>
    TensorView<dtype> ones(Backend& backend, const std::vector<uint>& shape) {
        return filled<dtype>(backend, shape, 1);
    }
    #define TEMPLATE(dtype) template TensorView<dtype> ones(Backend& backend, const std::vector<uint>& shape);
    #include "specialize/numeric.h"


    template <typename dtype>
    TensorView<dtype> reshape(const TensorView<dtype>& view, const std::vector<uint>& shape) {
        uint size = parse_shape(*view.get_backend(), shape);
        if (size != parse_shape(*view.get_backend(), view.get_shape())) {
            view.get_backend()->log("reshaped total size must be the same");
            view.get_backend()->abort();
        }

        return TensorView<dtype>(*view.get_backend(), shape, view.get_base());
    }
    #define TEMPLATE(dtype) template TensorView<dtype> reshape(const TensorView<dtype>& view, const std::vector<uint>& shape);
    #include "specialize/all.h"

    template <typename dtype>
    TensorView<dtype> ravel(const TensorView<dtype>& view) {
        return TensorView<dtype>(*view.get_backend(), {view.get_size()}, view.get_base());
    }
    #define TEMPLATE(dtype) template TensorView<dtype> ravel(const TensorView<dtype>& view);
    #include "specialize/all.h"

    template <typename dtype>
    TensorView<dtype>& flatten(TensorView<dtype>& view) {
        view.set_shape({view.get_size()});
        return view;
    }
    #define TEMPLATE(dtype) template TensorView<dtype>& flatten(TensorView<dtype>& view);
    #include "specialize/all.h"

    template <typename dtype>
    TensorView<dtype> transpose(const TensorView<dtype>& view, const std::vector<uint>& order) {
        if (order.size() != view.get_ndim()) {
            view.get_backend()->log("transposed order must have the same size as tensor shape (ndim)");
            view.get_backend()->abort();
        }
        
        std::vector<iint> strides(order.size());
        std::vector<uint> shape(order.size());
        std::vector<bool> used(order.size(), false);

        for (iint i = 0; i < order.size(); i++) {
            if (order[i] >= order.size()) {
                view.get_backend()->log("transposed order is invalid");
                view.get_backend()->abort();
            }
            if (used[order[i]]) {
                view.get_backend()->log("duplicated element in transposed order");
                view.get_backend()->abort();
            }

            used[order[i]] = true;
            strides[i] = view.get_strides()[order[i]];
            shape[i] = view.get_shape()[order[i]];
        }

        return TensorView<dtype>(*view.get_backend(), shape, view.get_base(), 0, strides);
    }
    #define TEMPLATE(dtype) template TensorView<dtype> transpose(const TensorView<dtype>& view, const std::vector<uint>& order);
    #include "specialize/all.h"


    // try using gpu for as_type???

    // template <typename dtype_old, typename dtype_new>
    // TensorView<dtype_new> as_type(const TensorView<dtype_old>& view) {
    //     return view;
    // }


}