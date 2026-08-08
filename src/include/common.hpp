#pragma once

#include <cstdint>
#include <string>
#include <format>


namespace oxide {
    

    using uint = unsigned int; // not an oxide data type
    class OxideError : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    // oxide data types
    typedef int32_t int32;
    typedef float float32;
    typedef uint32_t uint32;

    const uint MAXDIMS = 32;

    // misc functions
    template <typename d_type>
    std::string with_type(const std::string& name);
    std::string ansi(int code, const std::string& str);


}