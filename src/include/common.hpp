#pragma once

#include <cstdint>
#include <string>
#include <format>


namespace oxide {


    class oxide_error {
        std::string msg;

        public:
            oxide_error(std::string& _msg): msg(_msg) {}
            const char* what() const noexcept {
                return msg.data();
            }
    };

    // not oxide data types (used to make numeric sizes more predictable)
    using uint = uint32_t;
    using iint = int32_t;

    // oxide data types (things that can be stored in tensors)
    typedef int32_t int32;
    typedef float float32;

    const uint MAXDIMS = 16;
    const float MAXSEEDF = 4294967295.0f;
    const uint CACHETHRESHOLD = 1024 * 16;

    // misc functions
    template <typename dtype>
    std::string with_type(const std::string& name);
    std::string ansi(iint code, const std::string& str);


}