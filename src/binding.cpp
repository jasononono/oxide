#include <nanobind/nanobind.h>
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

        // m.def("rand", &oxide::rand,)
    }
 
    // oxide.hpp
    {
        m.def("free_backend", &oxide::free_backend, nb::arg("backend"));
        m.def("memory_optimize", &oxide::memory_optimize, nb::arg("backend"));
    }

    // tensor.hpp
    {
        auto c = nb::class_<oxide::TensorView<oxide::int32>>(m, "TensorView");
        c.def(nb::init<oxide::Backend&, const std::vector<uint>&, oxide::TensorData<oxide::int32>*>());
        c.def(nb::init<oxide::Backend&, const std::vector<uint>&, oxide::TensorData<oxide::int32>*, int, const std::vector<int>&>());
        c.def("get_element", &oxide::TensorView<oxide::int32>::get_element, nb::arg("indices"));
        c.def("set_element", &oxide::TensorView<oxide::int32>::set_element, nb::arg("indices"), nb::arg("value"));
        c.def("get_ndim", &oxide::TensorView<oxide::int32>::get_ndim);
        c.def("get_size", &oxide::TensorView<oxide::int32>::get_size);
        c.def("get_offset", &oxide::TensorView<oxide::int32>::get_offset);
        c.def("get_shape", &oxide::TensorView<oxide::int32>::get_shape);
        c.def("get_strides", 
        &oxide::TensorView<oxide::int32>::get_strides);
        c.def("get_string", &oxide::TensorView<oxide::int32>::get_string);

        auto c = nb::class_<oxide::TensorView<oxide::float32>>(m, "TensorView");
        c.def(nb::init<oxide::Backend&, const std::vector<uint>&, oxide::TensorData<oxide::float32>*>());
        c.def(nb::init<oxide::Backend&, const std::vector<uint>&, oxide::TensorData<oxide::float32>*, int, const std::vector<int>&>());
        c.def("get_element", &oxide::TensorView<oxide::float32>::get_element, nb::arg("indices"));
        c.def("set_element", &oxide::TensorView<oxide::float32>::set_element, nb::arg("indices"), nb::arg("value"));
        c.def("get_ndim", &oxide::TensorView<oxide::float32>::get_ndim);
        c.def("get_size", &oxide::TensorView<oxide::float32>::get_size);
        c.def("get_offset", &oxide::TensorView<oxide::float32>::get_offset);
        c.def("get_shape", &oxide::TensorView<oxide::float32>::get_shape);
        c.def("get_strides", 
        &oxide::TensorView<oxide::float32>::get_strides);
        c.def("get_string", &oxide::TensorView<oxide::float32>::get_string);
    }


}