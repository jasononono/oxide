#include <iostream>
#include "oxide.hpp"


namespace ox = oxide;


void run(ox::Backend& backend, ox::Dispatcher& dispatcher) {
    ox::TensorView<ox::float32> a = ox::random<ox::float32>(dispatcher, {2, 4}, 1, 2);
    ox::TensorView<ox::float32> b = ox::rand(dispatcher, {4});

    std::cout << a.get_string() << std::endl;
    std::cout << b.get_string() << std::endl;

    ox::usub<ox::float32>(dispatcher, a, b);

    std::cout << a.get_string() << std::endl;
}


int main() {
    ox::Backend backend;
    ox::Dispatcher dispatcher(backend);

    try {
        run(backend, dispatcher);
    } catch (const ox::oxide_error& e) {
        std::cout << ox::ansi(31, "Oxide: " + std::string(e.what())) << std::flush;
        return 1;
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }

    ox::free_backend(backend);
    return 0;
}