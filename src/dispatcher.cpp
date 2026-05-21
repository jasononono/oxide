#include "oxide.hpp"


namespace ox {


    Dispatcher::Dispatcher(Backend& _backend):
    backend(&_backend) {}

    Backend* Dispatcher::get_backend() const {
        return backend;
    }


    void Dispatcher::binary_operand(const std::string& function, int size, MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* out) {
        MTL::CommandBuffer* cmd = backend->new_cmd_buffer();
        MTL::ComputeCommandEncoder* encoder = cmd->computeCommandEncoder();
        if (!encoder) {
            backend->log("Oxide: failed to initialize command encoder."); backend->abort();
        }

        NS::UInteger max_threads = backend->set_cps(encoder, function);
        encoder->setBuffer(a, 0, 0);
        encoder->setBuffer(b, 0, 1);
        encoder->setBuffer(out, 0, 2);

        if (max_threads > size) {max_threads = size;}
        MTL::Size threads(max_threads, 1, 1);
        MTL::Size grid_size = MTL::Size(size, 1, 1);

        encoder->dispatchThreads(grid_size, threads);
        encoder->endEncoding();
        cmd->commit();
        cmd->waitUntilCompleted();

        encoder->release();
        cmd->release();
    }
    
    Tensor binary_add(Dispatcher& dispatcher, const Tensor& a, const Tensor& b) {
        if (a.get_backend() != b.get_backend() || a.get_backend() != dispatcher.get_backend()) {
            dispatcher.get_backend()->log("Oxide: backend mismatch");
            dispatcher.get_backend()->abort();
        }
        if (a.get_size() != b.get_size()) {
            dispatcher.get_backend()->log("Oxide: tensor sizes must be the same.");
            dispatcher.get_backend()->abort();
        }

        Tensor out(*dispatcher.get_backend(), a.get_size(), 0);
        dispatcher.binary_operand("add", a.get_size(), a.get_buffer(), b.get_buffer(), out.get_buffer());
        return out;
    }


}