#include "oxide.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include "string"


namespace nb = nanobind;


NB_MODULE(core, m) {


    // backend.hpp
    {
        auto c = nb::class_<oxide::Backend>(m, "Backend");
        c.def(nb::init());
    }

    // common.hpp
    {
        auto c = nb::class_<oxide::oxide_error>(m, "oxide_error");
        c.def("what", &oxide::oxide_error::what);
            
        m.def("ansi", &oxide::ansi, nb::arg("code"), nb::arg("str"));
    }

    // dispatcher.hpp
    {
        auto c = nb::class_<oxide::Dispatcher>(m, "Dispatcher");
        c.def(nb::init<oxide::Backend&>(), nb::arg("_backend"));
    }
    

    // functions.hpp
    {

        #define TEMPLATE(d_type) m.def("binary_add", &oxide::binary_add<oxide::d_type>, nb::arg("dispatcher"), nb::arg("a"), nb::arg("b"));
        #include "specialize/numeric.h"
        #define TEMPLATE(d_type) m.def("unary_add", &oxide::unary_add<oxide::d_type>, nb::arg("dispatcher"), nb::arg("a"), nb::arg("b"));
        #include "specialize/numeric.h"

        #define TEMPLATE(d_type) m.def("make_view", &oxide::make_view<oxide::d_type>, nb::arg("backend"), nb::arg("shape"), nb::arg("data"));
        #include "specialize/all.h"

        m.def("rand", &oxide::rand, nb::arg("dispatcher"), nb::arg("shape"));
        #define TEMPLATE(d_type) m.def("random", &oxide::random<oxide::d_type>, nb::arg("dispatcher"), nb::arg("shape"), nb::arg("a"), nb::arg("b"));
        #include "specialize/numeric.h"

        #define TEMPLATE(d_type) m.def("filled", &oxide::filled<oxide::d_type>, nb::arg("backend"), nb::arg("shape"), nb::arg("value"));
        #include "specialize/numeric.h"
        #define TEMPLATE(d_type) m.def("zeros", &oxide::zeros<oxide::d_type>, nb::arg("backend"), nb::arg("shape"));
        #include "specialize/numeric.h"
        #define TEMPLATE(d_type) m.def("ones", &oxide::ones<oxide::d_type>, nb::arg("backend"), nb::arg("shape"));
        #include "specialize/numeric.h"

        #define TEMPLATE(d_type) m.def("reshape", &oxide::reshape<oxide::d_type>, nb::arg("view"), nb::arg("shape"));
        #include "specialize/all.h"
        #define TEMPLATE(d_type) m.def("ravel", &oxide::ravel<oxide::d_type>, nb::arg("view"));
        #include "specialize/all.h"
        // #define TEMPLATE(d_type) m.def("flatten", &oxide::flatten<oxide::d_type>, nb::arg("view"));
        // #include "specialize/all.h"
        // TODO: figure out why tis thing doesnt work
        #define TEMPLATE(d_type) m.def("transpose", &oxide::transpose<oxide::d_type>, nb::arg("view"), nb::arg("order"));
        #include "specialize/all.h"
    }
 
    // oxide.hpp
    {
        m.def("free_backend", &oxide::free_backend, nb::arg("backend"));
        m.def("memory_optimize", &oxide::memory_optimize, nb::arg("backend"));
    }

    // tensor.hpp
    {
        #define TEMPLATE(d_type) \
        { \
            std::string name = std::string("TensorView_") + #d_type; \
            auto c = nb::class_<oxide::TensorView<oxide::d_type>>(m, name.data()); \
            c.def(nb::init<oxide::Backend&, const std::vector<uint>&, oxide::TensorData<oxide::d_type>*>()); \
            c.def(nb::init<oxide::Backend&, const std::vector<uint>&,  oxide::TensorData<oxide::d_type>*, oxide::iint, const std::vector<oxide::iint>&>()); \
            c.def("get_element", &oxide::TensorView<oxide::d_type>::get_element, nb::arg("indices")); \
            c.def("set_element", &oxide::TensorView<oxide::d_type>::set_element, nb::arg("indices"), nb::arg("value")); \
            c.def("get_ndim", &oxide::TensorView<oxide::d_type>::get_ndim); \
            c.def("get_size", &oxide::TensorView<oxide::d_type>::get_size); \
            c.def("get_offset", &oxide::TensorView<oxide::d_type>::get_offset); \
            c.def("get_shape", &oxide::TensorView<oxide::d_type>::get_shape); \
            c.def("get_strides", &oxide::TensorView<oxide::d_type>::get_strides); \
            c.def("get_string", &oxide::TensorView<oxide::d_type>::get_string); \
        }
        #include "specialize/all.h"
    }


}