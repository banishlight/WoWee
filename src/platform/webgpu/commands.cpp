// Command pools and command buffers.
//
// Recording only appends to the buffer's own list, which is plain C++ and
// safe on any thread - the renderer records its passes on worker threads.
// Nothing touches WebGPU until the list is replayed at vkQueueSubmit.

#include "wgpu_layer.hpp"

#include <algorithm>
#include <cstring>

using namespace wgpuvk;
namespace c = wgpuvk::cmd;

namespace {

template <typename T>
void record(VkCommandBuffer cb, T&& command) {
    cb->commands.emplace_back(std::forward<T>(command));
}

} // namespace

extern "C" {

VKAPI_ATTR VkResult VKAPI_CALL vkCreateCommandPool(
        VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*, VkCommandPool* pPool) {
    *pPool = new VkCommandPool_T();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyCommandPool(VkDevice, VkCommandPool pool, const VkAllocationCallbacks*) {
    if (!pool) return;
    releaseLater([pool] {
        for (auto* cb : pool->buffers) delete cb;
        delete pool;
    });
}

VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandPool(VkDevice, VkCommandPool pool, VkCommandPoolResetFlags) {
    for (auto* cb : pool->buffers) cb->commands.clear();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkAllocateCommandBuffers(
        VkDevice, const VkCommandBufferAllocateInfo* ai, VkCommandBuffer* pBuffers) {
    for (uint32_t i = 0; i < ai->commandBufferCount; ++i) {
        auto* cb = new VkCommandBuffer_T();
        cb->pool = ai->commandPool;
        ai->commandPool->buffers.push_back(cb);
        pBuffers[i] = cb;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkFreeCommandBuffers(
        VkDevice, VkCommandPool pool, uint32_t count, const VkCommandBuffer* buffers) {
    for (uint32_t i = 0; i < count; ++i) {
        VkCommandBuffer cb = buffers[i];
        if (!cb) continue;
        auto& v = pool->buffers;
        v.erase(std::remove(v.begin(), v.end(), cb), v.end());
        releaseLater([cb] { delete cb; });
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vkBeginCommandBuffer(VkCommandBuffer cb, const VkCommandBufferBeginInfo*) {
    cb->commands.clear();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEndCommandBuffer(VkCommandBuffer) {
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkResetCommandBuffer(VkCommandBuffer cb, VkCommandBufferResetFlags) {
    cb->commands.clear();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkCmdBeginRenderPass(
        VkCommandBuffer cb, const VkRenderPassBeginInfo* bi, VkSubpassContents) {
    record(cb, c::BeginRenderPass{bi->renderPass, bi->framebuffer, bi->renderArea,
                                  {bi->pClearValues, bi->pClearValues + bi->clearValueCount}});
}

VKAPI_ATTR void VKAPI_CALL vkCmdEndRenderPass(VkCommandBuffer cb) {
    record(cb, c::EndRenderPass{});
}

VKAPI_ATTR void VKAPI_CALL vkCmdBindPipeline(VkCommandBuffer cb, VkPipelineBindPoint, VkPipeline pipeline) {
    record(cb, c::BindPipeline{pipeline});
}

VKAPI_ATTR void VKAPI_CALL vkCmdBindDescriptorSets(
        VkCommandBuffer cb, VkPipelineBindPoint, VkPipelineLayout, uint32_t firstSet,
        uint32_t setCount, const VkDescriptorSet* sets, uint32_t dynamicCount, const uint32_t* dynamicOffsets) {
    record(cb, c::BindDescriptorSets{firstSet, {sets, sets + setCount},
                                     {dynamicOffsets, dynamicOffsets + dynamicCount}});
}

VKAPI_ATTR void VKAPI_CALL vkCmdBindVertexBuffers(
        VkCommandBuffer cb, uint32_t first, uint32_t count, const VkBuffer* buffers, const VkDeviceSize* offsets) {
    record(cb, c::BindVertexBuffers{first, {buffers, buffers + count}, {offsets, offsets + count}});
}

VKAPI_ATTR void VKAPI_CALL vkCmdBindIndexBuffer(
        VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkIndexType type) {
    record(cb, c::BindIndexBuffer{buffer, offset, type});
}

VKAPI_ATTR void VKAPI_CALL vkCmdPushConstants(
        VkCommandBuffer cb, VkPipelineLayout, VkShaderStageFlags, uint32_t offset, uint32_t size,
        const void* values) {
    const auto* p = static_cast<const uint8_t*>(values);
    record(cb, c::PushConstants{offset, {p, p + size}});
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetViewport(
        VkCommandBuffer cb, uint32_t, uint32_t count, const VkViewport* viewports) {
    if (count > 0) record(cb, c::SetViewport{viewports[0]});
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetScissor(
        VkCommandBuffer cb, uint32_t, uint32_t count, const VkRect2D* scissors) {
    if (count > 0) record(cb, c::SetScissor{scissors[0]});
}

VKAPI_ATTR void VKAPI_CALL vkCmdSetDepthBias(VkCommandBuffer cb, float constant, float clamp, float slope) {
    record(cb, c::SetDepthBias{constant, clamp, slope});
}

VKAPI_ATTR void VKAPI_CALL vkCmdDraw(
        VkCommandBuffer cb, uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex,
        uint32_t firstInstance) {
    record(cb, c::Draw{vertexCount, instanceCount, firstVertex, firstInstance});
}

VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexed(
        VkCommandBuffer cb, uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex,
        int32_t vertexOffset, uint32_t firstInstance) {
    record(cb, c::DrawIndexed{indexCount, instanceCount, firstIndex, vertexOffset, firstInstance});
}

VKAPI_ATTR void VKAPI_CALL vkCmdDrawIndexedIndirect(
        VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, uint32_t count, uint32_t stride) {
    record(cb, c::DrawIndexedIndirect{buffer, offset, count, stride});
}

VKAPI_ATTR void VKAPI_CALL vkCmdDispatch(VkCommandBuffer cb, uint32_t x, uint32_t y, uint32_t z) {
    record(cb, c::Dispatch{x, y, z});
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyBuffer(
        VkCommandBuffer cb, VkBuffer src, VkBuffer dst, uint32_t count, const VkBufferCopy* regions) {
    record(cb, c::CopyBuffer{src, dst, {regions, regions + count}});
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyBufferToImage(
        VkCommandBuffer cb, VkBuffer src, VkImage dst, VkImageLayout, uint32_t count,
        const VkBufferImageCopy* regions) {
    record(cb, c::CopyBufferToImage{src, dst, {regions, regions + count}});
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyImageToBuffer(
        VkCommandBuffer cb, VkImage src, VkImageLayout, VkBuffer dst, uint32_t count,
        const VkBufferImageCopy* regions) {
    record(cb, c::CopyImageToBuffer{src, dst, {regions, regions + count}});
}

VKAPI_ATTR void VKAPI_CALL vkCmdCopyImage(
        VkCommandBuffer cb, VkImage src, VkImageLayout, VkImage dst, VkImageLayout, uint32_t count,
        const VkImageCopy* regions) {
    record(cb, c::CopyImage{src, dst, {regions, regions + count}});
}

VKAPI_ATTR void VKAPI_CALL vkCmdBlitImage(
        VkCommandBuffer cb, VkImage src, VkImageLayout, VkImage dst, VkImageLayout, uint32_t count,
        const VkImageBlit* regions, VkFilter filter) {
    record(cb, c::BlitImage{src, dst, {regions, regions + count}, filter});
}

VKAPI_ATTR void VKAPI_CALL vkCmdClearColorImage(
        VkCommandBuffer cb, VkImage image, VkImageLayout, const VkClearColorValue* color, uint32_t count,
        const VkImageSubresourceRange* ranges) {
    record(cb, c::ClearColorImage{image, *color, {ranges, ranges + count}});
}

VKAPI_ATTR void VKAPI_CALL vkCmdClearDepthStencilImage(
        VkCommandBuffer cb, VkImage image, VkImageLayout, const VkClearDepthStencilValue* value,
        uint32_t count, const VkImageSubresourceRange* ranges) {
    record(cb, c::ClearDepthStencilImage{image, *value, {ranges, ranges + count}});
}

VKAPI_ATTR void VKAPI_CALL vkCmdFillBuffer(
        VkCommandBuffer cb, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, uint32_t data) {
    record(cb, c::FillBuffer{buffer, offset, size, data});
}

VKAPI_ATTR void VKAPI_CALL vkCmdExecuteCommands(
        VkCommandBuffer cb, uint32_t count, const VkCommandBuffer* buffers) {
    record(cb, c::ExecuteCommands{{buffers, buffers + count}});
}

// What WebGPU does for itself ---------------------------------------------------------------

VKAPI_ATTR void VKAPI_CALL vkCmdPipelineBarrier(
        VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags, uint32_t,
        const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*, uint32_t, const VkImageMemoryBarrier*) {}

VKAPI_ATTR void VKAPI_CALL vkCmdResetQueryPool(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t) {}

VKAPI_ATTR void VKAPI_CALL vkCmdWriteTimestamp(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, uint32_t) {}

// Submission ------------------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL vkQueueSubmit(
        VkQueue, uint32_t submitCount, const VkSubmitInfo* submits, VkFence fence) {
    std::vector<VkCommandBuffer_T*> buffers;
    for (uint32_t s = 0; s < submitCount; ++s) {
        for (uint32_t i = 0; i < submits[s].commandBufferCount; ++i)
            buffers.push_back(submits[s].pCommandBuffers[i]);
        for (uint32_t i = 0; i < submits[s].signalSemaphoreCount; ++i)
            submits[s].pSignalSemaphores[i]->value.fetch_add(1);
    }
    if (onMainThread()) {
        drainMainQueue();
        replay(buffers);
        if (fence) fence->signalled = true;
        return VK_SUCCESS;
    }
    // From a worker: the lists are copied now, because the worker may reset
    // and re-record its buffers as soon as it has waited on the fence, and
    // that wait only ends once the main thread has run them.
    auto copies = std::make_shared<std::vector<VkCommandBuffer_T>>();
    for (auto* cb : buffers) copies->push_back(*cb);
    runOnMain([copies, fence] {
        std::vector<VkCommandBuffer_T*> ptrs;
        for (auto& cb : *copies) ptrs.push_back(&cb);
        replay(ptrs);
        if (fence) fence->signalled = true;
    });
    return VK_SUCCESS;
}

// Dynamic rendering is never advertised, so these are never called; the
// renderer only links against them.
VKAPI_ATTR void VKAPI_CALL vkCmdBeginRendering(VkCommandBuffer, const VkRenderingInfo*) {}
VKAPI_ATTR void VKAPI_CALL vkCmdEndRendering(VkCommandBuffer) {}

} // extern "C"
