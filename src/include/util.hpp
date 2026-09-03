/*
UTIL.HPP

utility functions that are more complex and require other oxide definitions
*/


#pragma once

#include "common.hpp"
#include "backend.hpp"


namespace oxide {


    uint parse_shape(Backend& backend, const std::vector<uint>& shape); // throws error if shape is invalid, then returns accumulative size


}