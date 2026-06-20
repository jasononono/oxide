#pragma once

#include <cstdint>
#include <string>


namespace oxide {


    typedef int32_t int32;
    typedef float float32;

    const unsigned int MAXDIMS = 32;

    
    template <typename d_type>
    std::string with_type(const std::string& name);


}