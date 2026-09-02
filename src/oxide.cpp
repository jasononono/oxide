/*
OXIDE.CPP

implementation of oxide.hpp
*/


#include "oxide.hpp"


namespace oxide {


    void free_backend(Backend& backend) {
        TensorMemory key;
        
        while (!backend.get_tensors().empty()) {
            key = backend.get_tensors().back();
            free_mem(backend, key);
            backend.mem_delete(key);
        }
    }

    void free_mem(Backend& backend, TensorMemory memory) {
        // match type and delete
        #define TEMPLATE(dtype) \
        if (memory.tensor_type == typeid(TensorData<dtype>)) { \
            delete reinterpret_cast<TensorData<dtype>*>(memory.address); \
            return; \
        }
        #include "specialize/all.h"

        backend.log("tensor type is not recognized in memory, cannot free");
        backend.abort();
    }

    void mem_optimize(Backend& backend) {
        if (backend.mem_cacheget() < CACHETHRESHOLD) {return;}
        backend.mem_cacheclear();

        iint i = 0;
        TensorMemory key;

        while (i < backend.get_tensors().size()) {
            key = backend.get_tensors()[i];
            if (backend.get_mem_tied(key).empty()) { // if data is tied to no views
                free_mem(backend, key);
                backend.mem_delete(key);
            } else {
                i++;
            }
        }
    }


}