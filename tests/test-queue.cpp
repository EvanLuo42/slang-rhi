#include "testing.h"

using namespace rhi;
using namespace rhi::testing;

GPU_TEST_CASE("queue-graphics", ALL)
{
    auto queue = device->getQueue(QueueType::Graphics);
    REQUIRE(queue);
    CHECK(queue->getType() == QueueType::Graphics);
}

GPU_TEST_CASE("queue-availability", ALL)
{
    ComPtr<ICommandQueue> computeQueue;
    ComPtr<ICommandQueue> transferQueue;
    Result computeResult = device->getQueue(QueueType::Compute, computeQueue.writeRef());
    Result transferResult = device->getQueue(QueueType::Transfer, transferQueue.writeRef());

    if (device->hasFeature(Feature::ComputeQueue))
    {
        REQUIRE_CALL(computeResult);
        REQUIRE(computeQueue);
        CHECK(computeQueue->getType() == QueueType::Compute);
    }
    else
    {
        CHECK(computeResult == SLANG_E_NOT_AVAILABLE);
        CHECK(!computeQueue);
    }

    if (device->hasFeature(Feature::TransferQueue))
    {
        REQUIRE_CALL(transferResult);
        REQUIRE(transferQueue);
        CHECK(transferQueue->getType() == QueueType::Transfer);
    }
    else
    {
        CHECK(transferResult == SLANG_E_NOT_AVAILABLE);
        CHECK(!transferQueue);
    }
}

GPU_TEST_CASE("queue-compute-dispatch", ALL)
{
    if (!device->hasFeature(Feature::ComputeQueue))
    {
        SKIP("Compute queue not supported");
    }

    ComPtr<IShaderProgram> shaderProgram;
    REQUIRE_CALL(loadProgram(device, "test-compute-trivial", "computeMain", shaderProgram.writeRef()));

    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));

    const int numberCount = 4;
    float initialData[] = {0.0f, 1.0f, 2.0f, 3.0f};
    BufferDesc bufferDesc = {};
    bufferDesc.size = numberCount * sizeof(float);
    bufferDesc.elementSize = sizeof(float);
    bufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopySource;
    bufferDesc.defaultState = ResourceState::UnorderedAccess;
    bufferDesc.memoryType = MemoryType::DeviceLocal;

    ComPtr<IBuffer> buffer;
    REQUIRE_CALL(device->createBuffer(bufferDesc, initialData, buffer.writeRef()));

    auto queue = device->getQueue(QueueType::Compute);
    REQUIRE(queue);
    auto commandEncoder = queue->createCommandEncoder();
    auto passEncoder = commandEncoder->beginComputePass();
    REQUIRE(passEncoder);
    auto rootObject = passEncoder->bindPipeline(pipeline);
    ShaderCursor shaderCursor(rootObject);
    shaderCursor["buffer"].setBinding(buffer);
    float value = 10.f;
    shaderCursor["value"].setData(value);
    passEncoder->dispatchCompute(1, 1, 1);
    passEncoder->end();
    queue->submit(commandEncoder->finish());
    queue->waitOnHost();

    compareComputeResult(device, buffer, makeArray<float>(11.0f, 12.0f, 13.0f, 14.0f));
}

GPU_TEST_CASE("queue-transfer-copy", ALL)
{
    if (!device->hasFeature(Feature::TransferQueue))
    {
        SKIP("Transfer queue not supported");
    }

    const uint32_t count = 4;
    uint32_t initialData[] = {1, 2, 3, 4};

    BufferDesc srcDesc = {};
    srcDesc.size = count * sizeof(uint32_t);
    srcDesc.usage = BufferUsage::CopySource;
    srcDesc.defaultState = ResourceState::CopySource;
    srcDesc.memoryType = MemoryType::DeviceLocal;
    ComPtr<IBuffer> src;
    REQUIRE_CALL(device->createBuffer(srcDesc, initialData, src.writeRef()));

    BufferDesc dstDesc = {};
    dstDesc.size = count * sizeof(uint32_t);
    dstDesc.usage = BufferUsage::CopyDestination | BufferUsage::CopySource;
    dstDesc.defaultState = ResourceState::CopyDestination;
    dstDesc.memoryType = MemoryType::DeviceLocal;
    ComPtr<IBuffer> dst;
    REQUIRE_CALL(device->createBuffer(dstDesc, nullptr, dst.writeRef()));

    auto queue = device->getQueue(QueueType::Transfer);
    REQUIRE(queue);
    auto commandEncoder = queue->createCommandEncoder();
    commandEncoder->copyBuffer(dst, 0, src, 0, srcDesc.size);
    queue->submit(commandEncoder->finish());
    queue->waitOnHost();

    compareComputeResult(device, dst, makeArray<uint32_t>(1, 2, 3, 4));
}

GPU_TEST_CASE("queue-cross-queue-sync", ALL)
{
    if (!device->hasFeature(Feature::ComputeQueue))
    {
        SKIP("Compute queue not supported");
    }

    ComPtr<IShaderProgram> shaderProgram;
    REQUIRE_CALL(loadProgram(device, "test-compute-trivial", "computeMain", shaderProgram.writeRef()));

    ComputePipelineDesc pipelineDesc = {};
    pipelineDesc.program = shaderProgram.get();
    ComPtr<IComputePipeline> pipeline;
    REQUIRE_CALL(device->createComputePipeline(pipelineDesc, pipeline.writeRef()));

    const int numberCount = 4;
    float initialData[] = {0.0f, 1.0f, 2.0f, 3.0f};
    BufferDesc bufferDesc = {};
    bufferDesc.size = numberCount * sizeof(float);
    bufferDesc.elementSize = sizeof(float);
    bufferDesc.usage = BufferUsage::ShaderResource | BufferUsage::UnorderedAccess | BufferUsage::CopySource;
    bufferDesc.defaultState = ResourceState::UnorderedAccess;
    bufferDesc.memoryType = MemoryType::DeviceLocal;
    ComPtr<IBuffer> buffer;
    REQUIRE_CALL(device->createBuffer(bufferDesc, initialData, buffer.writeRef()));

    ComPtr<IFence> fence;
    FenceDesc fenceDesc = {};
    REQUIRE_CALL(device->createFence(fenceDesc, fence.writeRef()));

    auto computeQueue = device->getQueue(QueueType::Compute);
    auto graphicsQueue = device->getQueue(QueueType::Graphics);
    REQUIRE(computeQueue);
    REQUIRE(graphicsQueue);

    {
        auto commandEncoder = computeQueue->createCommandEncoder();
        auto passEncoder = commandEncoder->beginComputePass();
        auto rootObject = passEncoder->bindPipeline(pipeline);
        ShaderCursor shaderCursor(rootObject);
        shaderCursor["buffer"].setBinding(buffer);
        float value = 10.f;
        shaderCursor["value"].setData(value);
        passEncoder->dispatchCompute(1, 1, 1);
        passEncoder->end();

        ComPtr<ICommandBuffer> commandBuffer = commandEncoder->finish();
        SubmitDesc submitDesc = {};
        ICommandBuffer* commandBuffers[] = {commandBuffer.get()};
        IFence* signalFences[] = {fence.get()};
        uint64_t signalValues[] = {1};
        submitDesc.commandBuffers = commandBuffers;
        submitDesc.commandBufferCount = 1;
        submitDesc.signalFences = signalFences;
        submitDesc.signalFenceValues = signalValues;
        submitDesc.signalFenceCount = 1;
        REQUIRE_CALL(computeQueue->submit(submitDesc));
    }

    {
        SubmitDesc submitDesc = {};
        IFence* waitFences[] = {fence.get()};
        uint64_t waitValues[] = {1};
        submitDesc.waitFences = waitFences;
        submitDesc.waitFenceValues = waitValues;
        submitDesc.waitFenceCount = 1;
        REQUIRE_CALL(graphicsQueue->submit(submitDesc));
        REQUIRE_CALL(graphicsQueue->waitOnHost());
    }

    compareComputeResult(device, buffer, makeArray<float>(11.0f, 12.0f, 13.0f, 14.0f));
}
