#include <iostream>
#include "oxide.hpp"


void run() {
    oxide::Backend backend;
    oxide::Dispatcher dispatcher(backend);

    oxide::TensorView<oxide::float32> a = oxide::random<oxide::float32>(dispatcher, {2, 4}, 2, 10);
    oxide::TensorView<oxide::float32> b = oxide::random<oxide::float32>(dispatcher, {0, 4}, 2, 10);


    oxide::flatten<oxide::float32>(a);

    
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