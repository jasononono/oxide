#include "functions.hpp"
#include "typeutil.hpp"


namespace oxide {


    template <typename d_type>
    Tensor<d_type> binary_add(Dispatcher& dispatcher, const Tensor<d_type>& a, const Tensor<d_type>& b) {
        if (a.get_backend() != b.get_backend() || a.get_backend() != dispatcher.get_backend()) {
            dispatcher.get_backend()->log("Oxide: backend mismatch");
            dispatcher.get_backend()->abort();
        }
        if (a.get_size() != b.get_size()) {
            dispatcher.get_backend()->log("Oxide: tensor sizes must be the same.");
            dispatcher.get_backend()->abort();
        }

        Tensor<d_type> out(*dispatcher.get_backend(), a.get_size(), 0);
        dispatcher.binary_operand(with_type<d_type>("add"), a.get_size(), a.get_buffer(), b.get_buffer(), out.get_buffer());
        return out;
    }

    template Tensor<int32> binary_add(Dispatcher& dispatcher, const Tensor<int32>& a, const Tensor<int32>& b);
    template Tensor<float32> binary_add(Dispatcher& dispatcher, const Tensor<float32>& a, const Tensor<float32>& b);


    template <typename d_type>
    Tensor<d_type>& unary_add(Dispatcher& dispatcher, Tensor<d_type>& a, const Tensor<d_type>& b) {
        if (a.get_backend() != b.get_backend() || a.get_backend() != dispatcher.get_backend()) {
            dispatcher.get_backend()->log("Oxide: backend mismatch");
            dispatcher.get_backend()->abort();
        }
        if (a.get_size() != b.get_size()) {
            dispatcher.get_backend()->log("Oxide: tensor sizes must be the same.");
            dispatcher.get_backend()->abort();
        }

        dispatcher.unary_operand(with_type<d_type>("uadd"), a.get_size(), a.get_buffer(), b.get_buffer());
        return a;
    }

    template Tensor<int32>& unary_add(Dispatcher& dispatcher, Tensor<int32>& a, const Tensor<int32>& b);
    template Tensor<float32>& unary_add(Dispatcher& dispatcher, Tensor<float32>& a, const Tensor<float32>& b);


    Tensor<float32> rand_float(Backend& backend, int size) {
        Tensor<float32> out(backend, size, 0);
        backend.rand(out.get_ptr(), size);
        return out;
    }


}