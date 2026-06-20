#include <iostream>
#include "oxide.hpp"


void run() {
    oxide::Backend backend = oxide::new_backend();
    oxide::Dispatcher dispatcher(backend);

    oxide::TensorView<oxide::float32> a = oxide::filled<oxide::float32>(backend, {1, 3, 2}, 6.7);
    oxide::TensorView<oxide::float32> b = oxide::rand<oxide::float32>(backend, {2, 1, 2}, 0, 1);
    oxide::TensorView<oxide::float32> out = oxide::binary_add(dispatcher, a, b);

    std::cout << "a:   " << a.get_string() << std::endl;
    std::cout << "b:   " << b.get_string() << std::endl;
    std::cout << "out: " << out.get_string() << std::endl;

    // oxide::TensorView<oxide::float32> av = oxide::reshape(a, {5, 2});
    // std::cout << "av:  " << av.get_string() << std::endl;
    // oxide::TensorView<oxide::float32> bv = oxide::reshape(b, {10, 1});
    // std::cout << "bv:  " << bv.get_string() << std::endl;

    oxide::free_backend(backend);
}


int main() {
    try {
        run();
    } catch (const std::runtime_error& e) {
        std::cout << e.what() << std::flush;
        return 1;
    }
}