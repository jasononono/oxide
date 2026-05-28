#include <iostream>
#include "oxide.hpp"


void run() {
    oxide::Backend backend;
    oxide::Dispatcher dispatcher(backend);

    oxide::Tensor<float32> a = rand_float(backend, 10);
    oxide::Tensor<float32> b = rand_float(backend, 10);
    oxide::Tensor<float32> out = oxide::binary_add(dispatcher, a, b);

    std::cout << "a:   " << a.get_string() << std::endl;
    std::cout << "b:   " << b.get_string() << std::endl;
    std::cout << "out: " << out.get_string() << std::endl;

    oxide::TensorView<float32> av(backend, {2, 5}, &a);
    std::cout << av[{1, 4}] << std::endl;
    
}


int main() {
    try {
        run();
    } catch (const std::runtime_error& e) {
        std::cout << e.what() << std::flush;
        return 1;
    }
}