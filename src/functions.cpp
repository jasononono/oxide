#include "functions.hpp"
#include "typeutil.hpp"


namespace oxide {


    template <typename d_type>
    TensorView<d_type> binary_add(Dispatcher& dispatcher, const TensorView<d_type>& a, const TensorView<d_type>& b) {
        if (a.get_backend() != b.get_backend() || a.get_backend() != dispatcher.get_backend()) {
            dispatcher.get_backend()->log("Oxide: backend mismatch");
            dispatcher.get_backend()->abort();
        }
        if (a.get_base()->get_size() != b.get_base()->get_size()) {
            dispatcher.get_backend()->log("Oxide: tensor sizes must be the same");
            dispatcher.get_backend()->abort();
        }

        Tensor<d_type>* out = new Tensor<d_type>(*dispatcher.get_backend(), a.get_base()->get_size(), 0);
        dispatcher.binary_operand(with_type<d_type>("add"), a.get_base()->get_size(), a.get_base()->get_buffer(), b.get_base()->get_buffer(), out->get_buffer());
        return TensorView<d_type>(*dispatcher.get_backend(), a.get_shape(), out);
    }

    template TensorView<int32> binary_add(Dispatcher& dispatcher, const TensorView<int32>& a, const TensorView<int32>& b);
    template TensorView<float32> binary_add(Dispatcher& dispatcher, const TensorView<float32>& a, const TensorView<float32>& b);


    template <typename d_type>
    TensorView<d_type>& unary_add(Dispatcher& dispatcher, TensorView<d_type>& a, const TensorView<d_type>& b) {
        if (a.get_backend() != b.get_backend() || a.get_backend() != dispatcher.get_backend()) {
            dispatcher.get_backend()->log("Oxide: backend mismatch");
            dispatcher.get_backend()->abort();
        }
        if (a.get_base()->get_size() != b.get_base()->get_size()) {
            dispatcher.get_backend()->log("Oxide: tensor sizes must be the same.");
            dispatcher.get_backend()->abort();
        }

        dispatcher.unary_operand(with_type<d_type>("uadd"), a.get_base()->get_size(), a.get_base()->get_buffer(), b.get_base()->get_buffer());
        return a;
    }

    template TensorView<int32>& unary_add(Dispatcher& dispatcher, TensorView<int32>& a, const TensorView<int32>& b);
    template TensorView<float32>& unary_add(Dispatcher& dispatcher, TensorView<float32>& a, const TensorView<float32>& b);


    template <>
    TensorView<int32> rand(Backend& backend, const std::vector<unsigned int>& shape, int32 a, int32 b) {
        unsigned int size = parse_shape(backend, shape);
        Tensor<int32>* out = new Tensor<int32>(backend, size, int32());

        std::uniform_int_distribution<int32> dist(a, b);
        for (int i = 0; i < size; i++) {
            out->get_ptr()[i] = dist(backend.random_generate());
        }

        return TensorView<int32>(backend, shape, out);
    }
    template <>
    TensorView<float32> rand(Backend& backend, const std::vector<unsigned int>& shape, float32 a, float32 b) {
        unsigned int size = parse_shape(backend, shape);
        Tensor<float32>* out = new Tensor<float32>(backend, size, float32());

        std::uniform_real_distribution<float32> dist(a, b);
        for (int i = 0; i < size; i++) {
            out->get_ptr()[i] = dist(backend.random_generate());
        }

        return TensorView<float32>(backend, shape, out);
    }


    template <typename d_type>
    TensorView<d_type> filled(Backend& backend, const std::vector<unsigned int>& shape, d_type value) {
        unsigned int size = parse_shape(backend, shape);
        Tensor<d_type>* out = new Tensor<d_type>(backend, size, value);
        return TensorView<d_type>(backend, shape, out);
    }

    template TensorView<int32> filled(Backend& backend, const std::vector<unsigned int>& shape, int32 value);
    template TensorView<float32> filled(Backend& backend, const std::vector<unsigned int>& shape, float32 value);


    template <typename d_type>
    TensorView<d_type> zeros(Backend& backend, const std::vector<unsigned int>& shape) {
        return filled<d_type>(backend, shape, 0);
    }

    template TensorView<int32> zeros(Backend& backend, const std::vector<unsigned int>& shape);
    template TensorView<float32> zeros(Backend& backend, const std::vector<unsigned int>& shape);


    template <typename d_type>
    TensorView<d_type> ones(Backend& backend, const std::vector<unsigned int>& shape) {
        return filled<d_type>(backend, shape, 1);
    }

    template TensorView<int32> ones(Backend& backend, const std::vector<unsigned int>& shape);
    template TensorView<float32> ones(Backend& backend, const std::vector<unsigned int>& shape);


    template <typename d_type>
    TensorView<d_type> reshape(const TensorView<d_type>& view, const std::vector<unsigned int>& shape) {
        unsigned int size = parse_shape(*view.get_backend(), shape);
        if (size != parse_shape(*view.get_backend(), view.get_shape())) {
            view.get_backend()->log("Oxide: reshaped total size must be the same"); view.get_backend()->abort();
        }

        return TensorView<d_type>(*view.get_backend(), shape, view.get_base());
    }

    template TensorView<int32> reshape(const TensorView<int32>& view, const std::vector<unsigned int>& shape);
    template TensorView<float32> reshape(const TensorView<float32>& view, const std::vector<unsigned int>& shape);


}