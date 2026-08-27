/*
OXIDE.HPP

top level functions, mostly memory management things
called by Python implementation

to use the c++ interface of Oxide:
    1. include this file to get everything in the namespace 'oxide'
    2. create Oxide::Backend and Oxide::Dispatcher objects
    3. call oxide::free_backend() upon exit
    4. see docs/sample_cpp.cpp for reference
*/


#pragma once

#include "backend.hpp"
#include "common.hpp"
#include "dispatcher.hpp"
#include "functions.hpp"
#include "tensor.hpp"


namespace oxide {


    void free_backend(Backend& backend); // free all data
    void free_mem(Backend& backend, TensorMemory memory); // free a TensorData object
    void mem_optimize(Backend& backend); // free unused TensorData objects (those that have no views attached to them)


}