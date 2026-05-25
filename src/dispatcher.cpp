#include "dispatcher.hpp"


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

    
    void Dispatcher::unary_operand(const std::string& function, int size, MTL::Buffer* a, MTL::Buffer* b) {
        MTL::CommandBuffer* cmd = backend->new_cmd_buffer();
        MTL::ComputeCommandEncoder* encoder = cmd->computeCommandEncoder();
        if (!encoder) {
            backend->log("Oxide: failed to initialize command encoder."); backend->abort();
        }

        NS::UInteger max_threads = backend->set_cps(encoder, function);
        encoder->setBuffer(a, 0, 0);
        encoder->setBuffer(b, 0, 1);

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


}