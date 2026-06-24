#include "oxide.hpp"


namespace oxide {


    void free_backend(Backend& backend) {
        for (const TensorMemory memory : backend.get_tensors()) {
            free_tensor_memory(backend, memory);
        }
    }

    void free_tensor_memory(Backend& backend, TensorMemory memory) {
        if (memory.tensor_type == typeid(TensorData<int32>)) {
            delete static_cast<TensorData<int32>*>(memory.address);
        } else if (memory.tensor_type == typeid(TensorData<float32>)) {
            delete static_cast<TensorData<float32>*>(memory.address);
        } else {
            backend.log("Oxide: can only manually free TensorData specializations");
            backend.abort();
        }
    }

    void memory_optimize(Backend& backend) {
        int i = 0;
        TensorMemory key;

        while (i < backend.get_tensors().size()) {
            key = backend.get_tensors()[i];
            if (backend.memory_tied(key).empty()) {
                free_tensor_memory(backend, key);
                backend.memory_delete(key);
            } else {
                i++;
            }
        }
    }


}