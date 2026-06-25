#include "functions.hpp"
#include "typeutil.hpp"


namespace oxide {


    template <typename d_type>
    TensorView<d_type> binary_add(Dispatcher& dispatcher, const TensorView<d_type>& a, const TensorView<d_type>& b) {
        if (a.get_backend() != b.get_backend() || a.get_backend() != dispatcher.get_backend()) {
            dispatcher.get_backend()->log("Oxide: backend mismatch");
            dispatcher.get_backend()->abort();
        }
        a.check_base(); b.check_base();
        
        uint ndim = std::max(a.get_ndim(), b.get_ndim());
        std::vector<uint> out_shape(ndim);
        uint size = 1;
        std::vector<int> a_strides(ndim), b_strides(ndim);
        uint idx, a_idx, b_idx;
        
        for (int i = 0; i < ndim; i++) {
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
                dispatcher.get_backend()->log("Oxide: tensors cannot be broadcasted");
                dispatcher.get_backend()->abort();
            }

            size *= out_shape[idx];
        }

        TensorData<d_type>* out = new TensorData<d_type>(*dispatcher.get_backend(), size, 0);
        TensorView<d_type> view(*dispatcher.get_backend(), out_shape, out);

        dispatcher.binary_operand(with_type<d_type>("add"), view.get_size(), a.get_base()->get_buffer(), b.get_base()->get_buffer(), out->get_buffer(), ndim, a_strides, a.get_offset(), b_strides, b.get_offset(), view.get_strides());
        return view;
    }

    template TensorView<int32> binary_add(Dispatcher& dispatcher, const TensorView<int32>& a, const TensorView<int32>& b);
    template TensorView<float32> binary_add(Dispatcher& dispatcher, const TensorView<float32>& a, const TensorView<float32>& b);


    template <typename d_type>
    TensorView<d_type>& unary_add(Dispatcher& dispatcher, TensorView<d_type>& a, const TensorView<d_type>& b) {
        if (a.get_backend() != b.get_backend() || a.get_backend() != dispatcher.get_backend()) {
            dispatcher.get_backend()->log("Oxide: backend mismatch");
            dispatcher.get_backend()->abort();
        }
        a.check_base(); b.check_base();
        
        if (b.get_ndim() > a.get_ndim()) {
            dispatcher.get_backend()->log("Oxide: tensors cannot be broadcasted");
            dispatcher.get_backend()->abort();
        }

        std::vector<int> b_strides(a.get_ndim());
        uint idx, b_idx;

        
        for (int i = 0; i < a.get_ndim(); i++) {
            idx = a.get_ndim() - i - 1;
            b_idx = b.get_ndim() - i - 1;

            if (i >= b.get_ndim()) {
                b_strides[idx] = 0;
            } else if (b.get_shape()[b_idx] == 1) {
                b_strides[idx] = 0;
            } else if (a.get_shape()[idx] == b.get_shape()[b_idx]) {
                b_strides[idx] = b.get_strides()[b_idx];
            } else {
                dispatcher.get_backend()->log("Oxide: tensors cannot be broadcasted");
                dispatcher.get_backend()->abort();
            }
        }

        dispatcher.unary_operand(with_type<d_type>("uadd"), a.get_size(), a.get_base()->get_buffer(), b.get_base()->get_buffer(), a.get_ndim(), a.get_strides(), a.get_offset(), b_strides, b.get_offset());
        return a;
    }

    template TensorView<int32>& unary_add(Dispatcher& dispatcher, TensorView<int32>& a, const TensorView<int32>& b);
    template TensorView<float32>& unary_add(Dispatcher& dispatcher, TensorView<float32>& a, const TensorView<float32>& b);


    template <typename d_type>
    TensorView<d_type> rand(Backend& backend, const std::vector<uint>& shape, d_type a, d_type b) {
        backend.log("Oxide: rand() is not supported for this data type");
        backend.abort();
    }

    template <>
    TensorView<int32> rand(Backend& backend, const std::vector<uint>& shape, int32 a, int32 b) {
        uint size = parse_shape(backend, shape);
        TensorData<int32>* out = new TensorData<int32>(backend, size, int32());

        std::uniform_int_distribution<int32> dist(a, b);
        for (int i = 0; i < size; i++) {
            out->get_ptr()[i] = dist(backend.random_generate());
        }

        return TensorView<int32>(backend, shape, out);
    }
    template <>
    TensorView<float32> rand(Backend& backend, const std::vector<uint>& shape, float32 a, float32 b) {
        uint size = parse_shape(backend, shape);
        TensorData<float32>* out = new TensorData<float32>(backend, size, float32());

        std::uniform_real_distribution<float32> dist(a, b);
        for (int i = 0; i < size; i++) {
            out->get_ptr()[i] = dist(backend.random_generate());
        }

        return TensorView<float32>(backend, shape, out);
    }


    template <typename d_type>
    TensorView<d_type> filled(Backend& backend, const std::vector<uint>& shape, d_type value) {
        uint size = parse_shape(backend, shape);
        TensorData<d_type>* out = new TensorData<d_type>(backend, size, value);
        return TensorView<d_type>(backend, shape, out);
    }

    template TensorView<int32> filled(Backend& backend, const std::vector<uint>& shape, int32 value);
    template TensorView<float32> filled(Backend& backend, const std::vector<uint>& shape, float32 value);


    template <typename d_type>
    TensorView<d_type> zeros(Backend& backend, const std::vector<uint>& shape) {
        return filled<d_type>(backend, shape, 0);
    }

    template TensorView<int32> zeros(Backend& backend, const std::vector<uint>& shape);
    template TensorView<float32> zeros(Backend& backend, const std::vector<uint>& shape);


    template <typename d_type>
    TensorView<d_type> ones(Backend& backend, const std::vector<uint>& shape) {
        return filled<d_type>(backend, shape, 1);
    }

    template TensorView<int32> ones(Backend& backend, const std::vector<uint>& shape);
    template TensorView<float32> ones(Backend& backend, const std::vector<uint>& shape);


    template <typename d_type>
    TensorView<d_type> reshape(const TensorView<d_type>& view, const std::vector<uint>& shape) {
        uint size = parse_shape(*view.get_backend(), shape);
        if (size != parse_shape(*view.get_backend(), view.get_shape())) {
            view.get_backend()->log("Oxide: reshaped total size must be the same"); view.get_backend()->abort();
        }

        return TensorView<d_type>(*view.get_backend(), shape, view.get_base());
    }

    template TensorView<int32> reshape(const TensorView<int32>& view, const std::vector<uint>& shape);
    template TensorView<float32> reshape(const TensorView<float32>& view, const std::vector<uint>& shape);


    template <typename d_type>
    TensorView<d_type> ravel(const TensorView<d_type>& view) {
        return TensorView<d_type>(*view.get_backend(), {view.get_size()}, view.get_base());
    }

    template TensorView<int32> ravel(const TensorView<int32>& view);
    template TensorView<float32> ravel(const TensorView<float32>& view);


    template <typename d_type>
    TensorView<d_type>& flatten(TensorView<d_type>& view) {
        view.set_shape({view.get_size()});
        return view;
    }

    template TensorView<int32>& flatten(TensorView<int32>& view);
    template TensorView<float32>& flatten(TensorView<float32>& view);


    template <typename d_type>
    TensorView<d_type> transpose(const TensorView<d_type>& view, const std::vector<uint>& order) {
        if (order.size() != view.get_ndim()) {
            view.get_backend()->log("Oxide: transposed order must have the same size as tensor shape (ndim)");
            view.get_backend()->abort();
        }
        
        std::vector<int> strides(order.size());
        std::vector<uint> shape(order.size());
        std::vector<bool> used(order.size(), false);

        for (int i = 0; i < order.size(); i++) {
            if (order[i] >= order.size()) {
                view.get_backend()->log("Oxide: transposed order is invalid");
                view.get_backend()->abort();
            }
            if (used[order[i]]) {
                view.get_backend()->log("Oxide: duplicated element in transposed order");
                view.get_backend()->abort();
            }

            used[order[i]] = true;
            strides[i] = view.get_strides()[order[i]];
            shape[i] = view.get_shape()[order[i]];
        }

        return TensorView<d_type>(*view.get_backend(), shape, view.get_base(), 0, strides);
    }

    template TensorView<int32> transpose(const TensorView<int32>& view, const std::vector<uint>& order);
    template TensorView<float32> transpose(const TensorView<float32>& view, const std::vector<uint>& order);


    // try using gpu for as_type???

    // template <typename d_type_old, typename d_type_new>
    // TensorView<d_type_new> as_type(const TensorView<d_type_old>& view) {
    //     return view;
    // }


}