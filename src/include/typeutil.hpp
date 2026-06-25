#pragma once

#include <cstdint>
#include <string>


namespace oxide {


    using uint = unsigned int; // not an oxide data type

    typedef int32_t int32;
    typedef float float32;
    typedef uint32_t uint32;

    const uint MAXDIMS = 32;

    
    template <typename d_type>
    std::string with_type(const std::string& name);


}