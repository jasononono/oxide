#pragma once

#include <algorithm>
#include "typeutil.hpp"
#include "tensor.hpp"
#include "dispatcher.hpp"


namespace oxide {


    template <typename d_type>
    TensorView<d_type> binary_add(Dispatcher& dispatcher, const TensorView<d_type>& a, const TensorView<d_type>& b); // out = a + b

    template <typename d_type>
    TensorView<d_type>& unary_add(Dispatcher& dispatcher, TensorView<d_type>& a, const TensorView<d_type>& b); // a += b
    

    template <typename d_type>
    TensorView<d_type> make_view(Backend& backend, const std::vector<uint>& shape, const std::vector<d_type>& data);

    template <typename d_type>
    TensorView<d_type> rand(Backend& backend, const std::vector<uint>& shape, d_type a, d_type b); // generate tensor randomly filled with elements of range [a, b]

    template <typename d_type>
    TensorView<d_type> filled(Backend& backend, const std::vector<uint>& shape, d_type value); // generate tensor filled with value

    template <typename d_type>
    TensorView<d_type> zeros(Backend& backend, const std::vector<uint>& shape); // generate tensor filled with 0s

    template <typename d_type>
    TensorView<d_type> ones(Backend& backend, const std::vector<uint>& shape); // generate tensor filled with 1s


    template <typename d_type>
    TensorView<d_type> reshape(const TensorView<d_type>& view, const std::vector<uint>& shape);

    template <typename d_type>
    TensorView<d_type> ravel(const TensorView<d_type>& view);

    template <typename d_type>
    TensorView<d_type>& flatten(TensorView<d_type>& view);

    template <typename d_type>
    TensorView<d_type> transpose(const TensorView<d_type>& view, const std::vector<uint>& order);

    template <typename d_type>
    TensorView<d_type> slice(const TensorView<d_type>& view, const std::vector<std::pair<int, int>>& indices);

    
    template <typename d_type_old, typename d_type_new>
    TensorView<d_type_new> as_type(const TensorView<d_type_old>& view);


}