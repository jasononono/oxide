#include "common.hpp"


namespace oxide {


    #define TEMPLATE(dtype) template<> \
    std::string with_type<dtype>(const std::string& name) { \
        return name + "_" + #dtype; \
    }
    #include "specialize/all.h"

    std::string ansi(iint code, const std::string& str) {
        return std::format("\x1b[{}m{}\x1b[0m", code, str);
    }


}