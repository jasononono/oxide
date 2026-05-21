#include <iostream>
#include "src/oxide.hpp"


void run() {
    ox::Backend backend;
    ox::Dispatcher dispatcher(backend);

    ox::Tensor a = rand(backend, 10);
    ox::Tensor b = rand(backend, 10);
    ox::Tensor out = ox::binary_add(dispatcher, a, b);

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