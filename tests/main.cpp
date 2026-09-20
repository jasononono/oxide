#include <iostream>
#include "oxide.hpp"

namespace ox = oxide;


void run(ox::Backend& backend, ox::Dispatcher& dispatcher) {
    auto a = ox::constview<ox::float32>(backend, 3);
    auto b = ox::filled<ox::float32>(backend, {5, 2}, 1);

    std::cout << ox::add(dispatcher, a, b).get_string() << std::endl;
}


OXIDE_MAIN