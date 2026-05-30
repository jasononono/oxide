#include <iostream>
#include "oxide.hpp"


void run() {
    oxide::Backend backend;
    oxide::Dispatcher dispatcher(backend);

    oxide::TensorView<float32> a = oxide::rand<float32>(backend, {2, 5}, 0, 1);
    oxide::TensorView<float32> b = oxide::rand<float32>(backend, {2, 5}, 0, 1);
    oxide::TensorView<float32> out = oxide::binary_add(dispatcher, a, b);

    std::cout << "a:   " << a.get_string() << std::endl;
    std::cout << "b:   " << b.get_string() << std::endl;
    std::cout << "out: " << out.get_string() << std::endl;

    delete a.get_base();
    delete b.get_base();
    delete out.get_base(); // do auto memory deallocation later lol
    
}


int main() {
    try {
        run();
    } catch (const std::runtime_error& e) {
        std::cout << e.what() << std::flush;
        return 1;
    }
}