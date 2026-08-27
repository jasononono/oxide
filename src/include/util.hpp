/*
UTIL.HPP

utility functions that require other Oxide definitions
*/


#pragma once

#include "common.hpp"
#include "backend.hpp"


namespace oxide {


    uint parse_shape(Backend& backend, const std::vector<uint>& shape); // throws error if shape is invalid, then returns accumulative size


}