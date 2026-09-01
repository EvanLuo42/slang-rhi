#include "testing.h"

#include "core/platform.h"

#if SLANG_RHI_ENABLE_CUDA
#include <slang-rhi/cuda-driver-api.h>
#endif

#if SLANG_RHI_ENABLE_VULKAN
#include <vulkan/vulkan.h>
#endif

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

// Copy a UAV-default buffer on the transfer queue. D3D12 COPY lists and dedicated Vulkan
// transfer families cannot emit UAV/shader-stage barriers; if recording still restores
// defaultState, debug/validation layers should fail this test.
GPU_TEST_CASE("queue-transfer-copy-uav-default", D3D12 | Vulkan | Metal | CUDA | DontCacheDevice)
{
    if (!device->hasFeature(Feature::TransferQueue))
    {
        SKIP("Transfer queue not supported");
    }

    const uint32_t count = 4;
    uint32_t initialData[] = {1, 2, 3, 4};

    BufferDesc srcDesc = {};
    srcDesc.size = count * sizeof(uint32_t);
    srcDesc.usage = BufferUsage::UnorderedAccess | BufferUsage::CopySource;
    srcDesc.defaultState = ResourceState::UnorderedAccess;
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

#if SLANG_RHI_ENABLE_CUDA
GPU_TEST_CASE("queue-cuda-stream-non-blocking", CUDA)
{
    if (!device->hasFeature(Feature::ComputeQueue) && !device->hasFeature(Feature::TransferQueue))
    {
        SKIP("Compute/transfer queues not supported");
    }

    REQUIRE_CALL(device->setCudaContextCurrent());

    SharedLibraryHandle cudaLib = nullptr;
#if SLANG_WINDOWS_FAMILY
    REQUIRE_CALL(loadSharedLibrary("nvcuda.dll", cudaLib));
#else
    REQUIRE_CALL(loadSharedLibrary("libcuda.so.1", cudaLib));
#endif
    auto cuStreamGetFlagsFn =
        reinterpret_cast<CUresult (*)(CUstream, unsigned int*)>(findSymbolAddressByName(cudaLib, "cuStreamGetFlags"));
    REQUIRE(cuStreamGetFlagsFn);

    auto checkNonBlocking = [&](QueueType type)
    {
        ComPtr<ICommandQueue> queue;
        Result result = device->getQueue(type, queue.writeRef());
        if (SLANG_FAILED(result))
        {
            return;
        }
        NativeHandle handle = {};
        REQUIRE_CALL(queue->getNativeHandle(&handle));
        CHECK_EQ(handle.type, NativeHandleType::CUstream);
        unsigned int flags = 0;
        REQUIRE(cuStreamGetFlagsFn(reinterpret_cast<CUstream>(handle.value), &flags) == CUDA_SUCCESS);
        CHECK((flags & CU_STREAM_NON_BLOCKING) != 0);
    };

    if (device->hasFeature(Feature::ComputeQueue))
    {
        checkNonBlocking(QueueType::Compute);
    }
    if (device->hasFeature(Feature::TransferQueue))
    {
        checkNonBlocking(QueueType::Transfer);
    }
}
#endif

#if SLANG_RHI_ENABLE_VULKAN
// Mirrors DeviceImpl queue-family selection in src/vulkan/vk-device.cpp so the fallback
// can be unit-tested without depending on the local GPU's family layout.
struct VulkanQueueSlot
{
    int family = -1;
    int index = -1;
};

struct VulkanQueueSlots
{
    VulkanQueueSlot graphics;
    VulkanQueueSlot compute;
    VulkanQueueSlot transfer;
};

static bool vulkanQueueFamilySupportsCopy(VkQueueFlags flags)
{
    return (flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT)) != 0;
}

static VulkanQueueSlots selectVulkanQueueSlots(const VkQueueFamilyProperties* familyProps, uint32_t familyCount)
{
    VulkanQueueSlots slots = {};
    std::vector<uint32_t> queuesPerFamily(familyCount, 0);

    auto tryAlloc = [&](int family) -> int
    {
        if (family < 0 || uint32_t(family) >= familyCount)
            return -1;
        if (queuesPerFamily[family] >= familyProps[family].queueCount)
            return -1;
        return int(queuesPerFamily[family]++);
    };

    int graphicsFamily = -1;
    for (uint32_t i = 0; i < familyCount; ++i)
    {
        if (familyProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            graphicsFamily = int(i);
            break;
        }
    }
    int graphicsIndex = tryAlloc(graphicsFamily);
    slots.graphics = {graphicsFamily, graphicsIndex};

    int computeFamily = -1;
    for (uint32_t i = 0; i < familyCount; ++i)
    {
        if ((familyProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(familyProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT))
        {
            computeFamily = int(i);
            break;
        }
    }
    if (computeFamily < 0)
        computeFamily = graphicsFamily;
    int computeIndex = tryAlloc(computeFamily);
    slots.compute = {computeFamily, computeIndex};

    int transferFamily = -1;
    for (uint32_t i = 0; i < familyCount; ++i)
    {
        const VkQueueFlags flags = familyProps[i].queueFlags;
        if ((flags & VK_QUEUE_TRANSFER_BIT) && !(flags & VK_QUEUE_GRAPHICS_BIT) && !(flags & VK_QUEUE_COMPUTE_BIT))
        {
            transferFamily = int(i);
            break;
        }
    }
    if (transferFamily < 0)
    {
        for (uint32_t i = 0; i < familyCount; ++i)
        {
            if (queuesPerFamily[i] < familyProps[i].queueCount)
            {
                transferFamily = int(i);
                break;
            }
        }
    }
    int transferIndex = tryAlloc(transferFamily);
    slots.transfer = {transferFamily, transferIndex};
    return slots;
}

TEST_CASE("queue-vulkan-transfer-family-selection")
{
    SUBCASE("dedicated-transfer-family")
    {
        VkQueueFamilyProperties families[3] = {};
        families[0].queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
        families[0].queueCount = 1;
        families[1].queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
        families[1].queueCount = 1;
        families[2].queueFlags = VK_QUEUE_TRANSFER_BIT;
        families[2].queueCount = 1;

        VulkanQueueSlots slots = selectVulkanQueueSlots(families, 3);
        CHECK_EQ(slots.transfer.family, 2);
        CHECK_GE(slots.transfer.index, 0);
        CHECK(vulkanQueueFamilySupportsCopy(families[slots.transfer.family].queueFlags));
    }

    SUBCASE("fallback-to-graphics-extra-queue")
    {
        VkQueueFamilyProperties families[1] = {};
        families[0].queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
        families[0].queueCount = 3;

        VulkanQueueSlots slots = selectVulkanQueueSlots(families, 1);
        CHECK_EQ(slots.transfer.family, 0);
        CHECK_GE(slots.transfer.index, 0);
        CHECK(vulkanQueueFamilySupportsCopy(families[slots.transfer.family].queueFlags));
    }

    SUBCASE("fallback-must-not-pick-video-only-family")
    {
        VkQueueFamilyProperties families[3] = {};
        families[0].queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
        families[0].queueCount = 1;
        families[1].queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
        families[1].queueCount = 1;
        families[2].queueFlags = VK_QUEUE_VIDEO_DECODE_BIT_KHR;
        families[2].queueCount = 1;

        VulkanQueueSlots slots = selectVulkanQueueSlots(families, 3);
        if (slots.transfer.index >= 0)
        {
            CHECK(vulkanQueueFamilySupportsCopy(families[uint32_t(slots.transfer.family)].queueFlags));
        }
    }
}
#endif
