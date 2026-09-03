#include <iostream>
#include "oxide.hpp" // this header contains everything in the library

namespace ox = oxide;


// this function must be named 'run' and have these exact arguments.
void run(ox::Backend& backend, ox::Dispatcher& dispatcher) {

    // initialize tensor A as a 3x4 matrix of random 32 bit floats in the range [2, 8]
    auto A = ox::random<ox::float32>(dispatcher, {3, 4}, 2, 8);
    // initialize tensor B as a 3x4 matrix filled with the value 3.14 (as 32 bit floats)
    auto B = ox::filled<ox::float32>(backend, {3, 4}, 3.14);

    // display tensor A formatted nicely
    std::cout << A.get_string() << std::endl;
    // retrieve an element of tensor A using a vector of indices
    std::cout << A[{2, 0}] << std::endl;
    // you can also access more information about the view, a few examples are given below:
    std::cout << A.get_ndim() << " " << A.get_size() << std::endl; // # of dimensions, total # of elements
    for (int i : A.get_shape()) {std::cout << i << " ";} // shape of tensor A
    std::cout << std::endl << std::endl;

    // perform element-wise addition on A and B (out = a + b)
    auto OUT = ox::add<ox::float32>(dispatcher, A, B);
    std::cout << OUT.get_string() << std::endl;

    // multiply element-wise in place (a *= b)
    ox::umul<ox::float32>(dispatcher, A, B);
    std::cout << A.get_string() << std::endl;

}


// the following macro must placed after the run() function.
// do not define 'int main()' again!
OXIDE_MAIN