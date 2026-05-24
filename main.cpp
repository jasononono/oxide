#include <iostream>
#include "oxide.hpp"


void run() {
    ox::Backend backend;
    ox::Dispatcher dispatcher(backend);

    ox::Tensor<float32> a = rand_float(backend, 10);
    ox::Tensor<float32> b = rand_float(backend, 10);
    ox::Tensor<float32> out = ox::binary_add(dispatcher, a, b);

    std::cout << "a:   " << a.get_string() << std::endl;
    std::cout << "b:   " << b.get_string() << std::endl;
    std::cout << "out: " << out.get_string() << std::endl;
    
}


int main() {
    try {
        run();
    } catch (const std::runtime_error& e) {
        std::cout << e.what() << std::flush;
        return 1;
    }
}