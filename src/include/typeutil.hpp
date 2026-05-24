#pragma once

#include <cstdint>
#include <string>


typedef int32_t int32;
typedef float float32;


namespace ox {


    template <typename d_type>
    std::string with_type(const std::string& name);


}