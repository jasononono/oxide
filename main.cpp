#include <iostream>
#include "oxide.hpp"


void run() {
    oxide::Backend backend;
    oxide::Dispatcher dispatcher(backend);

    oxide::TensorView<oxide::float32> a = oxide::random<oxide::float32>(dispatcher, {2, 4}, 2, 10);


    std::cout << "a:   " << a.get_string() << std::endl;

    
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