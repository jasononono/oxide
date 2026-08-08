#include <iostream>
#include "oxide.hpp"


void run() {
    oxide::Backend backend;
    oxide::Dispatcher dispatcher(backend);

    oxide::TensorView<oxide::float32> a = oxide::rand<oxide::float32>(backend, {2, 0, 4}, 0, 1);
    oxide::TensorView<oxide::float32> b = oxide::make_view<oxide::float32>(backend, {2, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});


    std::cout << "a:   " << a.get_string() << std::endl;
    std::cout << "b:   " << b.get_string() << std::endl;

    
    oxide::free_backend(backend);
}


int main() {
    try {
        run();
    } catch (const oxide::OxideError& e) {
        std::cout << oxide::ansi(31, "Oxide: " + std::string(e.what())) << std::flush;
        return 1;
    }
    return 0;
}