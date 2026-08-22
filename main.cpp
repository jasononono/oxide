#include <iostream>
#include "oxide.hpp"


void run() {
    oxide::Backend backend;
    oxide::Dispatcher dispatcher(backend);
    
    oxide::TensorView<oxide::float32> a = oxide::random<oxide::float32>(dispatcher, {2, 4}, 1, 2);
    oxide::TensorView<oxide::float32> b = oxide::rand(dispatcher, {4});

    std::cout << a.get_string() << std::endl;
    std::cout << b.get_string() << std::endl;

    oxide::usub<oxide::float32>(dispatcher, a, b);

    std::cout << a.get_string() << std::endl;

    oxide::free_backend(backend);
}


int main() {
    try {
        run();
    } catch (const oxide::oxide_error& e) {
        std::cout << oxide::ansi(31, "Oxide: " + std::string(e.what())) << std::flush;
        return 1;
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
    return 0;
}