#pragma once

#include "tensor.hpp"
#include "dispatcher.hpp"
#include "backend.hpp"
#include "common.hpp"
#include <algorithm>


namespace oxide {


    template <typename dtype>
    TensorView<dtype> binary_add(Dispatcher& dispatcher, const TensorView<dtype>& a, const TensorView<dtype>& b); // out = a + b

    template <typename dtype>
    TensorView<dtype>& unary_add(Dispatcher& dispatcher, TensorView<dtype>& a, const TensorView<dtype>& b); // a += b
    

    template <typename dtype>
    TensorView<dtype> make_view(Backend& backend, const std::vector<uint>& shape, const std::vector<dtype>& data);


    TensorView<float32> rand(Dispatcher& dispatcher, const std::vector<uint>& shape);

    template <typename dtype>
    TensorView<dtype> random(Dispatcher& dispatcher, const std::vector<uint>& shape, dtype a, dtype b); // generate tensor randomly filled with elements of range [a, b]


    template <typename dtype>
    TensorView<dtype> filled(Backend& backend, const std::vector<uint>& shape, dtype value); // generate tensor filled with value

    template <typename dtype>
    TensorView<dtype> zeros(Backend& backend, const std::vector<uint>& shape); // generate tensor filled with 0s

    template <typename dtype>
    TensorView<dtype> ones(Backend& backend, const std::vector<uint>& shape); // generate tensor filled with 1s


    template <typename dtype>
    TensorView<dtype> reshape(const TensorView<dtype>& view, const std::vector<uint>& shape);

    template <typename dtype>
    TensorView<dtype> ravel(const TensorView<dtype>& view);

    template <typename dtype>
    TensorView<dtype>& flatten(TensorView<dtype>& view);

    template <typename dtype>
    TensorView<dtype> transpose(const TensorView<dtype>& view, const std::vector<uint>& order);

    // template <typename dtype>
    // TensorView<dtype> slice(const TensorView<dtype>& view, const std::vector<std::pair<int, int>>& indices);

    
    // template <typename dtype_old, typename dtype_new>
    // TensorView<dtype_new> as_type(const TensorView<dtype_old>& view);


}