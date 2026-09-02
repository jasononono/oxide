/*
UTIL.CPP

implementation of util.hpp
*/


#include "util.hpp"


namespace oxide {


    uint parse_shape(Backend& backend, const std::vector<uint>& shape) {
        if (shape.size() > MAXDIMS) {
            backend.log("tensor max dimensions exceeded");
            backend.abort();
        }
        if (shape.size() == 0) {
            backend.log("tensor shape must not be empty");
            backend.abort();
        }
        
        uint size = 1;
        for (uint i : shape) {
            if (i <= 0) {
                backend.log("dimension must be greater than 0");
                backend.abort();
            }
            size *= i;
        }
        return size;
    }


}