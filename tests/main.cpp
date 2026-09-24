#include <iostream>
#include "oxide.hpp"

namespace ox = oxide;


void run(ox::Backend& backend, ox::Dispatcher& dispatcher) {
    auto a = ox::rand(dispatcher, {2, 2, 2});
    std::cout << a.get_string() << std::endl;

    auto b = ox::transpose(a, {2, 1, 0});
    std::cout << b.get_string() << std::endl;

    auto c = ox::reshape<ox::float32>(dispatcher, b, {8});
    std::cout << c.get_string() << std::endl;
}


OXIDE_MAIN