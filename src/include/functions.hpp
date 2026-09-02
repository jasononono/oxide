/*
FUNCTIONS.HPP

L3 functions providing basic interface for specific computations (e.g. tensor binary addition).
calling them requires passing either the Dispatcher or Backend object (based on function type)
*/


#pragma once

#include "tensor.hpp"
#include "dispatcher.hpp"
#include "backend.hpp"
#include "common.hpp"
#include <algorithm>


namespace oxide {


    // broadcasting the smaller array to the bigger one:
    // 1. the arrays are aligned to the right
    // 2. axes x, y can be broadcasted if:
    //      - x = y, or
    //      - x = 1 or y = 1 (the single value will be broadcasted to every element of the other array)
    template <typename dtype>
    uint broadcast(Dispatcher& dispatcher, const TensorView<dtype>& a, const TensorView<dtype>& b, const uint ndim, std::vector<uint>& out_shape, std::vector<iint>& a_strides, std::vector<iint>& b_strides);

    // broadcasting an array b to a fixed sized array a
    // 1. b is aligned to the right and must not have more dims than a
    // 2. axes x of a and y of b can be broadcasted if:
    //      - x = y, or
    //      - y = 1
    template <typename dtype>
    void ubroadcast(Dispatcher& dispatcher, const TensorView<dtype>& a, const TensorView<dtype>& b, std::vector<iint>& b_strides);


    // initialize operations for every arithmetic function (+-*/) and for every numeric dtype

    #define TEMPLATE(dtype) \
    TEMPLATE2D(add, dtype) \
    TEMPLATE2D(sub, dtype) \
    TEMPLATE2D(mul, dtype) \
    TEMPLATE2D(div, dtype)

    // binary operations, e.g. out = a + b
    #define TEMPLATE2D(op, null) template <typename dtype> \
    TensorView<dtype> op(Dispatcher& dispatcher, const TensorView<dtype>& a, const TensorView<dtype>& b);
    TEMPLATE(null)
    #undef TEMPLATE2D

    // unary operations, e.g. a += b
    #define TEMPLATE2D(op, null) template <typename dtype> \
    TensorView<dtype>& u##op(Dispatcher& dispatcher, TensorView<dtype>& a, const TensorView<dtype>& b);
    TEMPLATE(null)
    #undef TEMPLATE2D

    #undef TEMPLATE


    template <typename dtype>
    TensorView<dtype> make_view(Backend& backend, const std::vector<uint>& shape, const std::vector<dtype>& data); // shortcut for initing from data & shape


    TensorView<float32> rand(Dispatcher& dispatcher, const std::vector<uint>& shape); // generate float32 tensor randomly filled with elements from range [0, 1)

    template <typename dtype>
    TensorView<dtype> random(Dispatcher& dispatcher, const std::vector<uint>& shape, dtype a, dtype b); // generate tensor randomly filled with elements from range [a, b]


    template <typename dtype>
    TensorView<dtype> filled(Backend& backend, const std::vector<uint>& shape, dtype value); // generate tensor filled with value

    template <typename dtype>
    TensorView<dtype> zeros(Backend& backend, const std::vector<uint>& shape); // generate tensor filled with 0s

    template <typename dtype>
    TensorView<dtype> ones(Backend& backend, const std::vector<uint>& shape); // generate tensor filled with 1s


    // these functions below do not modify the underlying data
    // a.k.a. they're fast

    template <typename dtype>
    TensorView<dtype> reshape(const TensorView<dtype>& view, const std::vector<uint>& shape); // returns a reshaped view without modifying the data

    template <typename dtype>
    TensorView<dtype> ravel(const TensorView<dtype>& view); // returns a flattened 1D view

    template <typename dtype>
    TensorView<dtype>& flatten(TensorView<dtype>& view); // what ravel() does, but modifies the view in-place

    template <typename dtype>
    TensorView<dtype> transpose(const TensorView<dtype>& view, const std::vector<uint>& order); // reorder the axes


    // i will implement these later lol (will i tho aaaaaa)

        // template <typename dtype>
        // TensorView<dtype> slice(const TensorView<dtype>& view, const std::vector<std::pair<int, int>>& indices);

        
        // template <typename dtype_old, typename dtype_new>
        // TensorView<dtype_new> as_type(const TensorView<dtype_old>& view);


}