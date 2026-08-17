#include <iostream>
#include "oxide.hpp"


void run() {
    oxide::Backend backend;
    oxide::Dispatcher dispatcher(backend);
    
    {
        oxide::TensorView<oxide::float32> a = oxide::zeros<oxide::float32>(backend, {256*16});
    }

    oxide::mem_optimize(backend);
    
    std::cout << backend.mem_cacheget() << std::endl;

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