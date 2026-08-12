#pragma once

#include <cstdint>
#include <string>
#include <format>


namespace oxide {


    class OxideError : public std::runtime_error {
        using std::runtime_error::runtime_error;
    };

    // not oxide data types (used to make numeric sizes more predictable)
    using uint = uint32_t;
    using iint = int32_t;

    // oxide data types (you can store these in tensors)
    typedef int32_t int32;
    typedef float float32;
    typedef uint32_t uint32;

    const uint MAXDIMS = 32;
    const float MAXSEEDF = 4294967295.0f;

    // misc functions
    template <typename d_type>
    std::string with_type(const std::string& name);
    std::string ansi(iint code, const std::string& str);


}