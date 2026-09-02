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


OXIDE_MAIN