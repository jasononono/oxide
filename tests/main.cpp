#include <iostream>
#include "oxide.hpp"

namespace ox = oxide;


void run(ox::Backend& backend, ox::Dispatcher& dispatcher) {
    auto a = ox::ones<ox::int32>(backend, {3, 4});
    std::cout << a.get_string() << std::endl;
    std::cout << a.contiguous() << std::endl;
    // auto b = ox::reshape<ox::int32>(dispatcher, a, {2, 6}, ox::AVOID);
    // std::cout << b.get_string() << std::endl;
}


OXIDE_MAIN