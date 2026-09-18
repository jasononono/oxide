/*
OXIDE.HPP

top level functions, mostly memory management things
called by Python implementation

to use the c++ interface of Oxide:
    1. include this file to get everything in the namespace 'oxide'
    2. use the OXIDE_MAIN macro defined below
    3. create a run() function with arguments oxide::Backend& and oxide::Dispatcher& (must be defined before using the macro)
    4. see docs/sample_cpp.cpp for reference
*/


#pragma once

#include "backend.hpp"
#include "common.hpp"
#include "dispatcher.hpp"
#include "functions.hpp"
#include "tensor.hpp"


namespace oxide {


    #define OXIDE_MAIN \
    void oxide_main() { \
        oxide::Backend backend; \
        backend.flush_warnings = true; \
        oxide::Dispatcher dispatcher(backend); \
        run(backend, dispatcher); \
        oxide::free_backend(backend); \
    } \
\
    int main() { \
        try { \
            oxide_main(); \
        } catch (const oxide::oxide_error& e) { \
            std::cout << oxide::ansi(31, "Oxide: " + std::string(e.what())) << std::flush; \
            return 1; \
        } catch (const std::exception& e) { \
            std::cout << e.what() << std::endl; \
            return 1; \
        } \
        return 0; \
    }


    void free_backend(Backend& backend); // free all data
    void free_mem(Backend& backend, TensorMemory memory); // free a TensorData object
    void mem_optimize(Backend& backend); // free unused TensorData objects (those that have no views attached to them)


}