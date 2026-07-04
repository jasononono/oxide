#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include "oxide.hpp"


namespace nb = nanobind;


NB_MODULE(core, m) {


    // backend.hpp
    {
        auto c = nb::class_<oxide::Backend>(m, "Backend");
        c.def(nb::init());
    }

    // dispatcher.hpp
    {
        auto c = nb::class_<oxide::Dispatcher>(m, "Dispatcher");
        c.def(nb::init<oxide::Backend&>(), nb::arg("_backend"));
    }
    

    // functions.hpp
    {
        m.def("binary_add", &oxide::binary_add<oxide::int32>, nb::arg("dispatcher"), nb::arg("a"), nb::arg("b"));
        m.def("binary_add", &oxide::binary_add<oxide::float32>, nb::arg("dispatcher"), nb::arg("a"), nb::arg("b"));
        // m.def("unary_add", &oxide::binary_add, nb::arg("dispatcher"), nb::arg('a'), nb::arg('b'));

        m.def("make_view", &oxide::make_view<oxide::int32>, nb::arg("backend"), nb::arg("shape"), nb::arg("data"));
        m.def("make_view", &oxide::make_view<oxide::float32>, nb::arg("backend"), nb::arg("shape"), nb::arg("data"));
        // m.def("rand", &oxide::rand,)
    }
 
    // oxide.hpp
    {
        m.def("free_backend", &oxide::free_backend, nb::arg("backend"));
        m.def("memory_optimize", &oxide::memory_optimize, nb::arg("backend"));
    }

    // tensor.hpp
    {
        auto c1 = nb::class_<oxide::TensorView<oxide::int32>>(m, "TensorView_int32");
        c1.def(nb::init<oxide::Backend&, const std::vector<uint>&, oxide::TensorData<oxide::int32>*>());
        c1.def(nb::init<oxide::Backend&, const std::vector<uint>&, oxide::TensorData<oxide::int32>*, int, const std::vector<int>&>());
        c1.def("get_element", &oxide::TensorView<oxide::int32>::get_element, nb::arg("indices"));
        c1.def("set_element", &oxide::TensorView<oxide::int32>::set_element, nb::arg("indices"), nb::arg("value"));
        c1.def("get_ndim", &oxide::TensorView<oxide::int32>::get_ndim);
        c1.def("get_size", &oxide::TensorView<oxide::int32>::get_size);
        c1.def("get_offset", &oxide::TensorView<oxide::int32>::get_offset);
        c1.def("get_shape", &oxide::TensorView<oxide::int32>::get_shape);
        c1.def("get_strides", 
        &oxide::TensorView<oxide::int32>::get_strides);
        c1.def("get_string", &oxide::TensorView<oxide::int32>::get_string);

        auto c2 = nb::class_<oxide::TensorView<oxide::float32>>(m, "TensorView_float32");
        c2.def(nb::init<oxide::Backend&, const std::vector<uint>&, oxide::TensorData<oxide::float32>*>());
        c2.def(nb::init<oxide::Backend&, const std::vector<uint>&, oxide::TensorData<oxide::float32>*, int, const std::vector<int>&>());
        c2.def("get_element", &oxide::TensorView<oxide::float32>::get_element, nb::arg("indices"));
        c2.def("set_element", &oxide::TensorView<oxide::float32>::set_element, nb::arg("indices"), nb::arg("value"));
        c2.def("get_ndim", &oxide::TensorView<oxide::float32>::get_ndim);
        c2.def("get_size", &oxide::TensorView<oxide::float32>::get_size);
        c2.def("get_offset", &oxide::TensorView<oxide::float32>::get_offset);
        c2.def("get_shape", &oxide::TensorView<oxide::float32>::get_shape);
        c2.def("get_strides", 
        &oxide::TensorView<oxide::float32>::get_strides);
        c2.def("get_string", &oxide::TensorView<oxide::float32>::get_string);
    }


}