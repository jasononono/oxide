/*
COMMON.HPP

misc functions, constants, and types that only need the stl
(utility instances that require other oxide definitions live in util.hpp)
*/


#pragma once

#include <cstdint>
#include <string>
#include <format>


namespace oxide {


    // custom error class (does not inherit off of std::runtime_error for now)
    class oxide_error {
        std::string msg;

        public:
            oxide_error(std::string& _msg): msg(_msg) {}
            const char* what() const noexcept {
                return msg.data();
            }
    };

    // not oxide data types (macros for implementation)
    using uint = uint32_t;
    using iint = int32_t;

    // oxide data types (things that can be stored in tensors)
    typedef int32_t int32;
    typedef float float32;

    const uint MAXDIMS = 16; // max tensor dimensions
    const float MAXSEEDF = 4294967295.0f; // maximum value of seed
    const uint CACHETHRESHOLD = 1024 * 16; // optimizes memory automatically when cache reaches this value

    // misc functions
    template <typename dtype>
    std::string with_type(const std::string& name); // returns name_dtype
    std::string ansi(iint code, const std::string& str);


}