#include "common.hpp"


namespace oxide {


    #define TEMPLATE(d_type) template<> \
    std::string with_type<d_type>(const std::string& name) { \
        return name + "_" + #d_type; \
    }
    #include "specialize/all.h"

    std::string ansi(int code, const std::string& str) {
        return std::format("\x1b[{}m{}\x1b[0m", code, str);
    }


}