#pragma once

#include "backend.hpp"
#include "dispatcher.hpp"
#include "functions.hpp"
#include "tensor.hpp"
#include "typeutil.hpp"


namespace oxide {


    Backend new_backend();
    void free_backend(Backend& backend);
    void free_tensor_memory(Backend& backend, TensorMemory& memory);
    void memory_optimize(Backend& backend);


}