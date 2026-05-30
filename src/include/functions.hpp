#pragma once

#include "typeutil.hpp"
#include "tensor.hpp"
#include "dispatcher.hpp"


namespace oxide {


    template <typename d_type>
    TensorView<d_type> binary_add(Dispatcher& dispatcher, const TensorView<d_type>& a, const TensorView<d_type>& b); // out = a + b

    template <typename d_type>
    TensorView<d_type>& unary_add(Dispatcher& dispatcher, TensorView<d_type>& a, const TensorView<d_type>& b); // a += b
    

    template <typename d_type>
    TensorView<d_type> rand(Backend& backend, const std::vector<unsigned int>& shape, d_type a, d_type b); // generate tensor randomly filled with elements of range [a, b]

    template <typename d_type>
    TensorView<d_type> filled(Backend& backend, const std::vector<unsigned int>& shape, d_type value); // generate tensor filled with value

    template <typename d_type>
    TensorView<d_type> zeros(Backend& backend, const std::vector<unsigned int>& shape); // generate tensor filled with 0s

    template <typename d_type>
    TensorView<d_type> ones(Backend& backend, const std::vector<unsigned int>& shape); // generate tensor filled with 1s


    template <typename d_type>
    TensorView<d_type> reshape(const TensorView<d_type>& view, const std::vector<unsigned int>& shape);

    template <typename d_type>
    TensorView<d_type> ravel(const TensorView<d_type>& view);

    template <typename d_type>
    TensorView<d_type> flatten(const TensorView<d_type>& view);

    template <typename d_type>
    TensorView<d_type> transpose(const TensorView<d_type>& view, const std::vector<unsigned int>& order);


}