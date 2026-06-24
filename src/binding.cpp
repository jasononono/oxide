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
        // m.def("binary_add", &oxide::binary_add, nb::arg("dispatcher"), nb::arg('a'), nb::arg('b'));
        // m.def("unary_add", &oxide::binary_add, nb::arg("dispatcher"), nb::arg('a'), nb::arg('b'));

        // m.def("rand", &oxide::rand,)
    }
 
    // oxide.hpp

    {
        m.def("free_backend", &oxide::free_backend, nb::arg("backend"));
        m.def("memory_optimize", &oxide::memory_optimize, nb::arg("backend"));
    }
    


}