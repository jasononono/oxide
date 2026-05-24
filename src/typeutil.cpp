#include "typeutil.hpp"


namespace ox {


    template<>
    std::string with_type<int32>(const std::string& name) {
        return name + "_int32";
    }
    template <>
    std::string with_type<float32>(const std::string& name) {
        return name + "_float32";
    }


}