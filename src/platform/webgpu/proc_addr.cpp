// vkGetInstanceProcAddr and vkGetDeviceProcAddr.
//
// vk-bootstrap loads everything it calls through these, and the renderer asks
// for its optional extension functions the same way. Anything not in the table
// answers nullptr, which every caller already treats as "not supported" - the
// debug-utils naming, device-fault reports and checkpoints among them.

#include "wgpu_layer.hpp"

#include <cstring>
#include <string_view>
#include <unordered_map>

#define WGPUVK_FUNCTIONS(X) \
    X(vkCreateInstance) X(vkDestroyInstance) X(vkEnumerateInstanceVersion) \
    X(vkEnumerateInstanceExtensionProperties) X(vkEnumerateInstanceLayerProperties) \
    X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceProperties2) \
    X(vkGetPhysicalDeviceFeatures) X(vkGetPhysicalDeviceFeatures2) \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceQueueFamilyProperties2) \
    X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceMemoryProperties2) \
    X(vkGetPhysicalDeviceFormatProperties) X(vkGetPhysicalDeviceFormatProperties2) \
    X(vkGetPhysicalDeviceImageFormatProperties) \
    X(vkEnumerateDeviceExtensionProperties) X(vkEnumerateDeviceLayerProperties) \
    X(vkCreateDevice) X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkGetDeviceQueue2) \
    X(vkDeviceWaitIdle) X(vkQueueWaitIdle) X(vkQueueSubmit) \
    X(vkCreateHeadlessSurfaceEXT) X(vkDestroySurfaceKHR) X(vkGetPhysicalDeviceSurfaceSupportKHR) \
    X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR) X(vkGetPhysicalDeviceSurfaceFormatsKHR) \
    X(vkGetPhysicalDeviceSurfacePresentModesKHR) X(vkCreateSwapchainKHR) X(vkDestroySwapchainKHR) \
    X(vkGetSwapchainImagesKHR) X(vkAcquireNextImageKHR) X(vkQueuePresentKHR) \
    X(vkCreateFence) X(vkDestroyFence) X(vkResetFences) X(vkGetFenceStatus) X(vkWaitForFences) \
    X(vkCreateSemaphore) X(vkDestroySemaphore) X(vkWaitSemaphores) \
    X(vkCreateQueryPool) X(vkDestroyQueryPool) X(vkGetQueryPoolResults) \
    X(vkCreatePipelineCache) X(vkDestroyPipelineCache) X(vkGetPipelineCacheData) \
    X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory) X(vkUnmapMemory) \
    X(vkFlushMappedMemoryRanges) X(vkInvalidateMappedMemoryRanges) \
    X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) \
    X(vkGetBufferMemoryRequirements2) X(vkGetDeviceBufferMemoryRequirements) \
    X(vkBindBufferMemory) X(vkBindBufferMemory2) \
    X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements) \
    X(vkGetImageMemoryRequirements2) X(vkGetDeviceImageMemoryRequirements) \
    X(vkBindImageMemory) X(vkBindImageMemory2) X(vkCreateImageView) X(vkDestroyImageView) \
    X(vkCreateSampler) X(vkDestroySampler) X(vkCreateShaderModule) X(vkDestroyShaderModule) \
    X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) X(vkCreateDescriptorPool) \
    X(vkDestroyDescriptorPool) X(vkResetDescriptorPool) X(vkAllocateDescriptorSets) \
    X(vkFreeDescriptorSets) X(vkUpdateDescriptorSets) \
    X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) \
    X(vkCreateRenderPass) X(vkCreateRenderPass2) X(vkDestroyRenderPass) \
    X(vkCreateFramebuffer) X(vkDestroyFramebuffer) \
    X(vkCreateGraphicsPipelines) X(vkCreateComputePipelines) X(vkDestroyPipeline) \
    X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkResetCommandPool) \
    X(vkAllocateCommandBuffers) X(vkFreeCommandBuffers) X(vkBeginCommandBuffer) \
    X(vkEndCommandBuffer) X(vkResetCommandBuffer) \
    X(vkCmdBeginRenderPass) X(vkCmdEndRenderPass) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) \
    X(vkCmdBindVertexBuffers) X(vkCmdBindIndexBuffer) X(vkCmdPushConstants) \
    X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdSetDepthBias) \
    X(vkCmdDraw) X(vkCmdDrawIndexed) X(vkCmdDrawIndexedIndirect) X(vkCmdDispatch) \
    X(vkCmdCopyBuffer) X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) X(vkCmdCopyImage) \
    X(vkCmdBlitImage) X(vkCmdClearColorImage) X(vkCmdClearDepthStencilImage) X(vkCmdFillBuffer) \
    X(vkCmdExecuteCommands) X(vkCmdPipelineBarrier) X(vkCmdResetQueryPool) X(vkCmdWriteTimestamp) \
    X(vkGetInstanceProcAddr) X(vkGetDeviceProcAddr)

namespace {

const std::unordered_map<std::string_view, PFN_vkVoidFunction>& table() {
    static const std::unordered_map<std::string_view, PFN_vkVoidFunction> t = {
#define WGPUVK_ENTRY(name) {#name, reinterpret_cast<PFN_vkVoidFunction>(&name)},
        WGPUVK_FUNCTIONS(WGPUVK_ENTRY)
#undef WGPUVK_ENTRY
        // The core-1.1 and KHR spellings vk-bootstrap and VMA may ask for.
        {"vkGetPhysicalDeviceProperties2KHR", reinterpret_cast<PFN_vkVoidFunction>(&vkGetPhysicalDeviceProperties2)},
        {"vkGetPhysicalDeviceFeatures2KHR", reinterpret_cast<PFN_vkVoidFunction>(&vkGetPhysicalDeviceFeatures2)},
        {"vkGetPhysicalDeviceMemoryProperties2KHR", reinterpret_cast<PFN_vkVoidFunction>(&vkGetPhysicalDeviceMemoryProperties2)},
        {"vkGetBufferMemoryRequirements2KHR", reinterpret_cast<PFN_vkVoidFunction>(&vkGetBufferMemoryRequirements2)},
        {"vkGetImageMemoryRequirements2KHR", reinterpret_cast<PFN_vkVoidFunction>(&vkGetImageMemoryRequirements2)},
        {"vkBindBufferMemory2KHR", reinterpret_cast<PFN_vkVoidFunction>(&vkBindBufferMemory2)},
        {"vkBindImageMemory2KHR", reinterpret_cast<PFN_vkVoidFunction>(&vkBindImageMemory2)},
        {"vkCreateRenderPass2KHR", reinterpret_cast<PFN_vkVoidFunction>(&vkCreateRenderPass2)},
        {"vkWaitSemaphoresKHR", reinterpret_cast<PFN_vkVoidFunction>(&vkWaitSemaphores)},
    };
    return t;
}

PFN_vkVoidFunction lookup(const char* name) {
    if (!name) return nullptr;
    auto it = table().find(name);
    return it == table().end() ? nullptr : it->second;
}

} // namespace

extern "C" {

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance, const char* pName) {
    return lookup(pName);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice, const char* pName) {
    return lookup(pName);
}

} // extern "C"
