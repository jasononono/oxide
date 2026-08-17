#pragma once

#include "backend.hpp"
#include "common.hpp"
#include "dispatcher.hpp"
#include "functions.hpp"
#include "tensor.hpp"


namespace oxide {


    void free_backend(Backend& backend);
    void free_mem(Backend& backend, TensorMemory memory);
    void mem_optimize(Backend& backend);


}