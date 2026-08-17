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
        c.def("mem_cacheget", &oxide::Backend::mem_cacheget);
    }

    // common.hpp
    {
        auto c = nb::exception<oxide::oxide_error>(m, "oxide_error");
            
        m.def("ansi", &oxide::ansi, nb::arg("code"), nb::arg("str"));
    }

    // dispatcher.hpp
    {
        auto c = nb::class_<oxide::Dispatcher>(m, "Dispatcher");
        c.def(nb::init<oxide::Backend&>(), nb::arg("_backend"));
    }
    

    // functions.hpp
    {

        #define TEMPLATE(dtype) m.def("add", &oxide::add<oxide::dtype>, nb::arg("dispatcher"), nb::arg("a"), nb::arg("b"));
        #include "specialize/numeric.h"
        #define TEMPLATE(dtype) m.def("uadd", &oxide::uadd<oxide::dtype>, nb::arg("dispatcher"), nb::arg("a"), nb::arg("b"));
        #include "specialize/numeric.h"

        m.def("rand", &oxide::rand, nb::arg("dispatcher"), nb::arg("shape"));
        #define TEMPLATE(dtype) m.def("random", &oxide::random<oxide::dtype>, nb::arg("dispatcher"), nb::arg("shape"), nb::arg("a"), nb::arg("b"));
        #include "specialize/numeric.h"

        #define TEMPLATE(dtype) m.def("filled", &oxide::filled<oxide::dtype>, nb::arg("backend"), nb::arg("shape"), nb::arg("value"));
        #include "specialize/numeric.h"
        #define TEMPLATE(dtype) m.def("zeros", &oxide::zeros<oxide::dtype>, nb::arg("backend"), nb::arg("shape"));
        #include "specialize/numeric.h"
        #define TEMPLATE(dtype) m.def("ones", &oxide::ones<oxide::dtype>, nb::arg("backend"), nb::arg("shape"));
        #include "specialize/numeric.h"

        #define TEMPLATE(dtype) m.def("reshape", &oxide::reshape<oxide::dtype>, nb::arg("view"), nb::arg("shape"));
        #include "specialize/all.h"
        #define TEMPLATE(dtype) m.def("ravel", &oxide::ravel<oxide::dtype>, nb::arg("view"));
        #include "specialize/all.h"
        #define TEMPLATE(dtype) m.def("flatten", &oxide::flatten<oxide::dtype>, nb::arg("view"));
        #include "specialize/all.h"
        #define TEMPLATE(dtype) m.def("transpose", &oxide::transpose<oxide::dtype>, nb::arg("view"), nb::arg("order"));
        #include "specialize/all.h"
    }
 
    // oxide.hpp
    {
        m.def("free_backend", &oxide::free_backend, nb::arg("backend"));
        m.def("mem_optimize", &oxide::mem_optimize, nb::arg("backend"));
    }

    // tensor.hpp
    {
        #define TEMPLATE(dtype) \
        { \
            std::string name = std::string("TensorView_") + #dtype; \
            auto c = nb::class_<oxide::TensorView<oxide::dtype>>(m, name.data()); \
            c.def(nb::init<oxide::Backend&, const std::vector<uint>&, oxide::TensorData<oxide::dtype>*>()); \
            c.def(nb::init<oxide::Backend&, const std::vector<uint>&,  oxide::TensorData<oxide::dtype>*, oxide::iint, const std::vector<oxide::iint>&>()); \
            c.def("get_element", &oxide::TensorView<oxide::dtype>::get_element, nb::arg("indices")); \
            c.def("set_element", &oxide::TensorView<oxide::dtype>::set_element, nb::arg("indices"), nb::arg("value")); \
            c.def("get_ndim", &oxide::TensorView<oxide::dtype>::get_ndim); \
            c.def("get_size", &oxide::TensorView<oxide::dtype>::get_size); \
            c.def("get_offset", &oxide::TensorView<oxide::dtype>::get_offset); \
            c.def("get_shape", &oxide::TensorView<oxide::dtype>::get_shape); \
            c.def("get_strides", &oxide::TensorView<oxide::dtype>::get_strides); \
            c.def("get_string", &oxide::TensorView<oxide::dtype>::get_string); \
        }
        #include "specialize/all.h"
    }


}