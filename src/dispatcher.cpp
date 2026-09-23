/*
DISPATCHER.CPP

implementation of dispatcher.hpp
*/


#include "dispatcher.hpp"


namespace oxide {


    Dispatcher::Dispatcher(Backend& _backend):
    backend(&_backend) {}

    Dispatcher::~Dispatcher() {}

    Backend* Dispatcher::get_backend() const {
        return backend;
    }

    
    // all of the functions below basically follows the same procedure:
    // 1. get a command buffer and encoder & set the function
    // 2. set the function arguments using designated buffer indices
    // 3. calculate grid size (how many threads to dispatch)
    // 4. dispatch, wait for completion, release

    void Dispatcher::binary_operation(const std::string& function, uint size, MTL::Buffer* a, MTL::Buffer* b, MTL::Buffer* out, uint ndim, const std::vector<iint>& a_strides, uint a_offset, const std::vector<iint>& b_strides, uint b_offset, const std::vector<iint>& out_strides) {
        MTL::CommandBuffer* cmd = backend->new_cmd_buffer();
        MTL::ComputeCommandEncoder* encoder = cmd->computeCommandEncoder();
        if (!encoder) {
            backend->log("failed to initialize command encoder.");
            backend->abort();
        }

        NS::UInteger max_threads = backend->set_cps(encoder, function);
        encoder->setBuffer(a, 0, 0);
        encoder->setBuffer(b, 0, 1);
        encoder->setBuffer(out, 0, 2);
        encoder->setBytes(&ndim, sizeof(uint), 3);
        encoder->setBytes(a_strides.data(), a_strides.size() * sizeof(uint), 4);
        encoder->setBytes(&a_offset, sizeof(uint), 5);
        encoder->setBytes(b_strides.data(), b_strides.size() * sizeof(uint), 6);
        encoder->setBytes(&b_offset, sizeof(uint), 7);
        encoder->setBytes(out_strides.data(), out_strides.size() * sizeof(uint), 8);

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

    void Dispatcher::unary_operation(const std::string& function, uint size, MTL::Buffer* a, MTL::Buffer* b, uint ndim, const std::vector<iint>& a_strides, uint a_offset, const std::vector<iint>& b_strides, uint b_offset) {
        MTL::CommandBuffer* cmd = backend->new_cmd_buffer();
        MTL::ComputeCommandEncoder* encoder = cmd->computeCommandEncoder();
        if (!encoder) {
            backend->log("failed to initialize command encoder.");
            backend->abort();
        }

        NS::UInteger max_threads = backend->set_cps(encoder, function);
        encoder->setBuffer(a, 0, 0);
        encoder->setBuffer(b, 0, 1);
        encoder->setBytes(&ndim, sizeof(uint), 2);
        encoder->setBytes(a_strides.data(), a_strides.size() * sizeof(uint), 3);
        encoder->setBytes(&a_offset, sizeof(uint), 4);
        encoder->setBytes(b_strides.data(), b_strides.size() * sizeof(uint), 5);
        encoder->setBytes(&b_offset, sizeof(uint), 6);

        size -= a_offset;
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


    void Dispatcher::rand(uint size, MTL::Buffer* buf, uint seed) {
        MTL::CommandBuffer* cmd = backend->new_cmd_buffer();
        MTL::ComputeCommandEncoder* encoder = cmd->computeCommandEncoder();
        if (!encoder) {
            backend->log("failed to initialize command encoder.");
            backend->abort();
        }

        NS::UInteger max_threads = backend->set_cps(encoder, "rand");
        encoder->setBuffer(buf, 0, 0);
        encoder->setBytes(&seed, sizeof(uint), 1);

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

    template <typename dtype>
    void Dispatcher::random(const std::string& function, uint size, MTL::Buffer* buf, uint seed, dtype a, dtype b) {
        MTL::CommandBuffer* cmd = backend->new_cmd_buffer();
        MTL::ComputeCommandEncoder* encoder = cmd->computeCommandEncoder();
        if (!encoder) {
            backend->log("failed to initialize command encoder.");
            backend->abort();
        }

        NS::UInteger max_threads = backend->set_cps(encoder, function);
        encoder->setBuffer(buf, 0, 0);
        encoder->setBytes(&seed, sizeof(uint), 1);
        encoder->setBytes(&a, sizeof(dtype), 2);
        encoder->setBytes(&b, sizeof(dtype), 3);

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
    #define TEMPLATE(dtype) template void Dispatcher::random(const std::string& function, uint size, MTL::Buffer* buf, uint seed, dtype a, dtype b);
    #include "specialize/numeric.h"


    void Dispatcher::memcpy(uint size, MTL::Buffer* a, MTL::Buffer* b, uint ndim, const std::vector<iint>& a_strides, uint a_offset, const std::vector<iint>& b_strides, uint b_offset) {

    }
    

}