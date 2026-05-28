#pragma once

#include "typeutil.hpp"
#include "tensor.hpp"
#include "dispatcher.hpp"


namespace oxide {


    template <typename d_type>
    Tensor<d_type> binary_add(Dispatcher& dispatcher, const Tensor<d_type>& a, const Tensor<d_type>& b);

    template <typename d_type>
    Tensor<d_type>& unary_add(Dispatcher& dispatcher, Tensor<d_type>& a, const Tensor<d_type>& b);
    

    Tensor<float32> rand_float(Backend& backend, int size); // generate tensor filled with random floats


}