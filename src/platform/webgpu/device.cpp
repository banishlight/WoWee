// Instance, device, queue, surface, swapchain and synchronisation.
//
// There is one of everything: one instance, one physical device standing for
// the GPU the page was given, one device with one queue that does everything.
// What the renderer and vk-bootstrap ask of them is answered from WebGPU's
// limits and features, so a query reports what the browser can really do.

#include "wgpu_layer.hpp"

#include <emscripten/emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/threading.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <thread>

#include "core/logger.hpp"

extern "C" WGPUDevice emscripten_webgpu_get_device(void);

namespace wgpuvk {

namespace {

VkInstance_T gInstance;
VkPhysicalDevice_T gPhysicalDevice;
VkDevice_T* gDevice = nullptr;

std::mutex gMainMutex;
std::vector<std::function<void()>> gMainQueue;

// Device-local memory, then host-visible memory twice: coherent, and not.
// A host-visible buffer's GPU copy is brought up to date from its CPU copy
// when a submission uses it. Coherent memory is what small buffers get and
// is uploaded whole; large buffers are only offered the non-coherent type,
// so the renderer's flushes say which ranges changed and only those move
// (see kWholeUploadMax). The non-coherent type is also HOST_CACHED, which is
// what VMA looks for in readback memory.
constexpr VkDeviceSize kDeviceHeap = 2048ull * 1024 * 1024;
constexpr VkDeviceSize kHostHeap = 512ull * 1024 * 1024;

void fillFeatures(VkPhysicalDeviceFeatures& f) {
    f = {};
    f.robustBufferAccess = VK_TRUE;
    f.fullDrawIndexUint32 = VK_TRUE;
    f.independentBlend = VK_TRUE;
    f.multiDrawIndirect = VK_TRUE;
    f.drawIndirectFirstInstance = VK_TRUE;
    f.depthBiasClamp = VK_TRUE;
    f.samplerAnisotropy = VK_TRUE;
    f.textureCompressionBC = gpu().bc ? VK_TRUE : VK_FALSE;
    f.fragmentStoresAndAtomics = VK_TRUE;
    f.shaderClipDistance = VK_FALSE;
    f.imageCubeArray = VK_TRUE;
}

/// Zeroes every struct in a pNext chain past its header, so an extension
/// struct this layer knows nothing of reads as "not supported".
void clearChain(void* pNext) {
    struct Header { VkStructureType sType; void* pNext; };
    for (auto* h = static_cast<Header*>(pNext); h; h = static_cast<Header*>(h->pNext)) {
        size_t size = 0;
        switch (h->sType) {
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES:
            size = sizeof(VkPhysicalDeviceVulkan11Features); break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES:
            size = sizeof(VkPhysicalDeviceVulkan12Features); break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES:
            size = sizeof(VkPhysicalDeviceVulkan13Features); break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES:
            size = sizeof(VkPhysicalDeviceVulkan11Properties); break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES:
            size = sizeof(VkPhysicalDeviceVulkan12Properties); break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES:
            size = sizeof(VkPhysicalDeviceVulkan13Properties); break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES:
            size = sizeof(VkPhysicalDeviceDepthStencilResolveProperties); break;
        case VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES:
            size = sizeof(VkPhysicalDeviceDriverProperties); break;
        default:
            // Unknown structs keep whatever the caller put in them; a caller
            // zero-initialises what it asks about, which reads as unsupported.
            continue;
        }
        std::memset(reinterpret_cast<uint8_t*>(h) + sizeof(Header), 0, size - sizeof(Header));
    }
}

template <typename T>
VkResult fillArray(uint32_t* count, T* out, const T* items, uint32_t n) {
    if (!out) { *count = n; return VK_SUCCESS; }
    const uint32_t written = std::min(*count, n);
    for (uint32_t i = 0; i < written; ++i) out[i] = items[i];
    *count = written;
    return written < n ? VK_INCOMPLETE : VK_SUCCESS;
}

void canvasSize(const std::string& canvas, uint32_t& w, uint32_t& h) {
    int cw = 0, ch = 0;
    emscripten_get_canvas_element_size(canvas.c_str(), &cw, &ch);
    w = static_cast<uint32_t>(std::max(cw, 1));
    h = static_cast<uint32_t>(std::max(ch, 1));
}

} // namespace

Gpu& gpu() {
    static Gpu g = [] {
        Gpu x;
        x.instance = wgpuCreateInstance(nullptr);
        x.device = emscripten_webgpu_get_device();
        if (!x.device) {
            LOG_FATAL("WebGPU: the page did not provide a device (webgpu_pre.js)");
            return x;
        }
        x.queue = wgpuDeviceGetQueue(x.device);
        wgpuDeviceGetLimits(x.device, &x.limits);
        x.bc = wgpuDeviceHasFeature(x.device, WGPUFeatureName_TextureCompressionBC);
        x.float32Filterable = wgpuDeviceHasFeature(x.device, WGPUFeatureName_Float32Filterable);
        x.depth32Stencil8 = wgpuDeviceHasFeature(x.device, WGPUFeatureName_Depth32FloatStencil8);
        x.multiDraw = wgpuDeviceHasFeature(x.device, WGPUFeatureName_MultiDrawIndirect);
        LOG_WARNING("WebGPU device: BC textures ", x.bc ? "yes" : "NO",
                    ", float32-filterable ", x.float32Filterable ? "yes" : "no",
                    ", depth32float-stencil8 ", x.depth32Stencil8 ? "yes" : "no",
                    ", multi-draw indirect ", x.multiDraw ? "yes" : "no");
        return x;
    }();
    return g;
}

bool onMainThread() {
    return emscripten_is_main_browser_thread();
}

void runOnMain(std::function<void()> fn) {
    if (onMainThread()) {
        fn();
        return;
    }
    std::lock_guard<std::mutex> lock(gMainMutex);
    gMainQueue.push_back(std::move(fn));
}

void releaseLater(std::function<void()> fn) {
    runOnMain(std::move(fn));
}

void drainMainQueue() {
    std::vector<std::function<void()>> work;
    {
        std::lock_guard<std::mutex> lock(gMainMutex);
        work.swap(gMainQueue);
    }
    for (auto& fn : work) fn();
}

} // namespace wgpuvk

using namespace wgpuvk;

extern "C" {

// Instance -----------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceVersion(uint32_t* pApiVersion) {
    *pApiVersion = VK_API_VERSION_1_2;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(
        const char* pLayerName, uint32_t* pPropertyCount, VkExtensionProperties* pProperties) {
    if (pLayerName) { *pPropertyCount = 0; return VK_SUCCESS; }
    static const VkExtensionProperties exts[] = {
        {VK_KHR_SURFACE_EXTENSION_NAME, 25},
        {VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME, 1},
        {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME, 2},
    };
    return fillArray(pPropertyCount, pProperties, exts, 3);
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(
        uint32_t* pPropertyCount, VkLayerProperties*) {
    *pPropertyCount = 0;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(
        const VkInstanceCreateInfo*, const VkAllocationCallbacks*, VkInstance* pInstance) {
    if (!gpu().device) return VK_ERROR_INITIALIZATION_FAILED;
    *pInstance = &gInstance;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyInstance(VkInstance, const VkAllocationCallbacks*) {}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDevices(
        VkInstance, uint32_t* pCount, VkPhysicalDevice* pDevices) {
    VkPhysicalDevice one = &gPhysicalDevice;
    return fillArray(pCount, pDevices, &one, 1);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties(
        VkPhysicalDevice, VkPhysicalDeviceProperties* p) {
    const WGPULimits& l = gpu().limits;
    *p = {};
    p->apiVersion = VK_API_VERSION_1_2;
    p->driverVersion = 1;
    p->deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    std::strncpy(p->deviceName, "WebGPU", VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1);
    VkPhysicalDeviceLimits& v = p->limits;
    v.maxImageDimension1D = l.maxTextureDimension1D;
    v.maxImageDimension2D = l.maxTextureDimension2D;
    v.maxImageDimension3D = l.maxTextureDimension3D;
    v.maxImageDimensionCube = l.maxTextureDimension2D;
    v.maxImageArrayLayers = l.maxTextureArrayLayers;
    v.maxUniformBufferRange = static_cast<uint32_t>(
        std::min<uint64_t>(l.maxUniformBufferBindingSize, UINT32_MAX));
    v.maxStorageBufferRange = static_cast<uint32_t>(
        std::min<uint64_t>(l.maxStorageBufferBindingSize, UINT32_MAX));
    v.maxPushConstantsSize = 128;
    v.maxMemoryAllocationCount = 1u << 20;
    v.maxSamplerAllocationCount = 4000;
    v.bufferImageGranularity = 1;
    v.maxBoundDescriptorSets = std::min(l.maxBindGroups, kMaxBoundSets);
    v.maxPerStageDescriptorSamplers = l.maxSamplersPerShaderStage;
    v.maxPerStageDescriptorUniformBuffers = l.maxUniformBuffersPerShaderStage;
    v.maxPerStageDescriptorStorageBuffers = l.maxStorageBuffersPerShaderStage;
    v.maxPerStageDescriptorSampledImages = l.maxSampledTexturesPerShaderStage;
    v.maxPerStageDescriptorStorageImages = l.maxStorageTexturesPerShaderStage;
    v.maxPerStageResources = 128;
    v.maxDescriptorSetSamplers = l.maxSamplersPerShaderStage * 4;
    v.maxDescriptorSetUniformBuffers = l.maxUniformBuffersPerShaderStage * 4;
    v.maxDescriptorSetUniformBuffersDynamic = l.maxDynamicUniformBuffersPerPipelineLayout;
    v.maxDescriptorSetStorageBuffers = l.maxStorageBuffersPerShaderStage * 4;
    v.maxDescriptorSetStorageBuffersDynamic = l.maxDynamicStorageBuffersPerPipelineLayout;
    v.maxDescriptorSetSampledImages = l.maxSampledTexturesPerShaderStage * 4;
    v.maxDescriptorSetStorageImages = l.maxStorageTexturesPerShaderStage * 4;
    v.maxVertexInputAttributes = l.maxVertexAttributes;
    v.maxVertexInputBindings = l.maxVertexBuffers;
    v.maxVertexInputAttributeOffset = 2047;
    v.maxVertexInputBindingStride = l.maxVertexBufferArrayStride;
    v.maxVertexOutputComponents = l.maxInterStageShaderVariables * 4;
    v.maxFragmentInputComponents = l.maxInterStageShaderVariables * 4;
    v.maxFragmentOutputAttachments = l.maxColorAttachments;
    v.maxColorAttachments = l.maxColorAttachments;
    v.maxComputeSharedMemorySize = l.maxComputeWorkgroupStorageSize;
    v.maxComputeWorkGroupCount[0] = v.maxComputeWorkGroupCount[1] =
        v.maxComputeWorkGroupCount[2] = l.maxComputeWorkgroupsPerDimension;
    v.maxComputeWorkGroupInvocations = l.maxComputeInvocationsPerWorkgroup;
    v.maxComputeWorkGroupSize[0] = l.maxComputeWorkgroupSizeX;
    v.maxComputeWorkGroupSize[1] = l.maxComputeWorkgroupSizeY;
    v.maxComputeWorkGroupSize[2] = l.maxComputeWorkgroupSizeZ;
    v.maxDrawIndexedIndexValue = UINT32_MAX;
    v.maxDrawIndirectCount = UINT32_MAX;
    v.maxSamplerLodBias = 0.0f;
    v.maxSamplerAnisotropy = 16.0f;
    v.maxViewports = 1;
    v.maxViewportDimensions[0] = v.maxViewportDimensions[1] = l.maxTextureDimension2D;
    v.viewportBoundsRange[0] = -2.0f * l.maxTextureDimension2D;
    v.viewportBoundsRange[1] = 2.0f * l.maxTextureDimension2D;
    v.minMemoryMapAlignment = 64;
    v.minTexelBufferOffsetAlignment = 256;
    v.minUniformBufferOffsetAlignment = l.minUniformBufferOffsetAlignment;
    v.minStorageBufferOffsetAlignment = l.minStorageBufferOffsetAlignment;
    v.maxFramebufferWidth = v.maxFramebufferHeight = l.maxTextureDimension2D;
    v.maxFramebufferLayers = 1;
    // WebGPU renders with one sample or four, and nothing else.
    const VkSampleCountFlags samples = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT;
    v.framebufferColorSampleCounts = samples;
    v.framebufferDepthSampleCounts = samples;
    v.framebufferStencilSampleCounts = samples;
    v.framebufferNoAttachmentsSampleCounts = samples;
    v.sampledImageColorSampleCounts = samples;
    v.sampledImageDepthSampleCounts = samples;
    v.sampledImageIntegerSampleCounts = VK_SAMPLE_COUNT_1_BIT;
    v.sampledImageStencilSampleCounts = samples;
    v.storageImageSampleCounts = VK_SAMPLE_COUNT_1_BIT;
    v.maxSampleMaskWords = 1;
    v.timestampComputeAndGraphics = VK_FALSE;
    v.timestampPeriod = 1.0f;
    v.maxClipDistances = 0;
    v.maxCullDistances = 0;
    v.discreteQueuePriorities = 2;
    v.pointSizeRange[0] = v.pointSizeRange[1] = 1.0f;
    v.lineWidthRange[0] = v.lineWidthRange[1] = 1.0f;
    v.optimalBufferCopyOffsetAlignment = 4;
    v.optimalBufferCopyRowPitchAlignment = 256;
    v.nonCoherentAtomSize = 4;
    p->sparseProperties = {};
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceProperties2(
        VkPhysicalDevice pd, VkPhysicalDeviceProperties2* p) {
    vkGetPhysicalDeviceProperties(pd, &p->properties);
    clearChain(p->pNext);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures(
        VkPhysicalDevice, VkPhysicalDeviceFeatures* f) {
    fillFeatures(*f);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFeatures2(
        VkPhysicalDevice, VkPhysicalDeviceFeatures2* f) {
    fillFeatures(f->features);
    // Nothing beyond 1.0 is offered: no timeline semaphores, no
    // synchronization2, no dynamic rendering. The renderer has a path for the
    // absence of each, and they are the simpler paths to translate.
    clearChain(f->pNext);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties(
        VkPhysicalDevice, uint32_t* pCount, VkQueueFamilyProperties* pProps) {
    VkQueueFamilyProperties family{};
    family.queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    family.queueCount = 1;
    family.timestampValidBits = 0;
    family.minImageTransferGranularity = {1, 1, 1};
    fillArray(pCount, pProps, &family, 1);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceQueueFamilyProperties2(
        VkPhysicalDevice pd, uint32_t* pCount, VkQueueFamilyProperties2* pProps) {
    if (!pProps) { vkGetPhysicalDeviceQueueFamilyProperties(pd, pCount, nullptr); return; }
    VkQueueFamilyProperties family{};
    uint32_t one = 1;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &one, &family);
    if (*pCount >= 1) { pProps[0].queueFamilyProperties = family; *pCount = 1; }
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties(
        VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* p) {
    *p = {};
    p->memoryHeapCount = 2;
    p->memoryHeaps[0] = {kDeviceHeap, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    p->memoryHeaps[1] = {kHostHeap, 0};
    p->memoryTypeCount = 3;
    p->memoryTypes[kMemDeviceLocal] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0};
    p->memoryTypes[kMemHostCoherent] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, 1};
    p->memoryTypes[kMemHostNonCoherent] = {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 1};
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceMemoryProperties2(
        VkPhysicalDevice pd, VkPhysicalDeviceMemoryProperties2* p) {
    vkGetPhysicalDeviceMemoryProperties(pd, &p->memoryProperties);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties(
        VkPhysicalDevice, VkFormat format, VkFormatProperties* p) {
    *p = {};
    if (vertexFormat(format) != WGPUVertexFormat_Force32) {
        p->bufferFeatures = VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT;
    }
    const FormatInfo info = formatInfo(format);
    if (info.wgpu == WGPUTextureFormat_Undefined) return;
    VkFormatFeatureFlags f = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
                             VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                             VK_FORMAT_FEATURE_TRANSFER_DST_BIT |
                             VK_FORMAT_FEATURE_BLIT_SRC_BIT;
    const bool isFloat32 = format == VK_FORMAT_R32_SFLOAT || format == VK_FORMAT_R32G32_SFLOAT ||
                           format == VK_FORMAT_R32G32B32A32_SFLOAT;
    const bool isInteger = format == VK_FORMAT_R8_UINT || format == VK_FORMAT_R16_UINT ||
                           format == VK_FORMAT_R32_UINT || format == VK_FORMAT_R16G16_UINT ||
                           format == VK_FORMAT_R8G8B8A8_UINT || format == VK_FORMAT_R32_SINT;
    if (!info.depth && !isInteger && (!isFloat32 || gpu().float32Filterable)) {
        f |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    }
    if (info.depth) {
        f |= VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
    } else if (info.blockDim == 1 && format != VK_FORMAT_R16_SNORM) {
        f |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
        if (!isInteger) f |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT;
        if (format == VK_FORMAT_R32_SFLOAT || format == VK_FORMAT_R32_UINT ||
            format == VK_FORMAT_R16G16B16A16_SFLOAT || format == VK_FORMAT_R8G8B8A8_UNORM ||
            format == VK_FORMAT_R32G32B32A32_SFLOAT || format == VK_FORMAT_R32G32_SFLOAT) {
            f |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
        }
    }
    p->optimalTilingFeatures = f;
    p->linearTilingFeatures = f & ~(VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                                   VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

VKAPI_ATTR void VKAPI_CALL vkGetPhysicalDeviceFormatProperties2(
        VkPhysicalDevice pd, VkFormat format, VkFormatProperties2* p) {
    vkGetPhysicalDeviceFormatProperties(pd, format, &p->formatProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceImageFormatProperties(
        VkPhysicalDevice, VkFormat format, VkImageType, VkImageTiling, VkImageUsageFlags,
        VkImageCreateFlags, VkImageFormatProperties* p) {
    if (formatInfo(format).wgpu == WGPUTextureFormat_Undefined) return VK_ERROR_FORMAT_NOT_SUPPORTED;
    const uint32_t dim = gpu().limits.maxTextureDimension2D;
    p->maxExtent = {dim, dim, gpu().limits.maxTextureDimension3D};
    p->maxMipLevels = 16;
    p->maxArrayLayers = gpu().limits.maxTextureArrayLayers;
    p->sampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT;
    p->maxResourceSize = 1ull << 31;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(
        VkPhysicalDevice, const char* pLayerName, uint32_t* pCount, VkExtensionProperties* pProps) {
    if (pLayerName) { *pCount = 0; return VK_SUCCESS; }
    static const VkExtensionProperties exts[] = {{VK_KHR_SWAPCHAIN_EXTENSION_NAME, 70}};
    return fillArray(pCount, pProps, exts, 1);
}

VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(
        VkPhysicalDevice, uint32_t* pCount, VkLayerProperties*) {
    *pCount = 0;
    return VK_SUCCESS;
}

// Device ----------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(
        VkPhysicalDevice, const VkDeviceCreateInfo*, const VkAllocationCallbacks*, VkDevice* pDevice) {
    if (!gDevice) gDevice = new VkDevice_T();
    *pDevice = gDevice;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDevice(VkDevice, const VkAllocationCallbacks*) {}

VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue(VkDevice device, uint32_t, uint32_t, VkQueue* pQueue) {
    *pQueue = &device->queue;
}

VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue2(VkDevice device, const VkDeviceQueueInfo2*, VkQueue* pQueue) {
    *pQueue = &device->queue;
}

VKAPI_ATTR VkResult VKAPI_CALL vkDeviceWaitIdle(VkDevice) {
    // Work is on the queue the moment it is submitted, and the queue finishes
    // it in order before anything later can see its results. What is left is
    // what worker threads handed to the main thread.
    if (onMainThread()) drainMainQueue();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkQueueWaitIdle(VkQueue) {
    if (onMainThread()) drainMainQueue();
    return VK_SUCCESS;
}

// Surface and swapchain --------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL vkCreateHeadlessSurfaceEXT(
        VkInstance, const VkHeadlessSurfaceCreateInfoEXT*, const VkAllocationCallbacks*,
        VkSurfaceKHR* pSurface) {
    // "Headless" in name only: the surface is the page's canvas, which is the
    // one SDL draws into. There is no Vulkan surface extension for a canvas,
    // and this is the one that asks nothing of the platform.
    auto* s = new VkSurfaceKHR_T();
    s->canvas = "#canvas";
    WGPUEmscriptenSurfaceSourceCanvasHTMLSelector canvas = WGPU_EMSCRIPTEN_SURFACE_SOURCE_CANVAS_HTML_SELECTOR_INIT;
    canvas.selector = {s->canvas.c_str(), s->canvas.size()};
    WGPUSurfaceDescriptor desc = WGPU_SURFACE_DESCRIPTOR_INIT;
    desc.nextInChain = &canvas.chain;
    s->surface = wgpuInstanceCreateSurface(gpu().instance, &desc);
    if (!s->surface) {
        delete s;
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    *pSurface = s;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroySurfaceKHR(VkInstance, VkSurfaceKHR surface, const VkAllocationCallbacks*) {
    if (!surface) return;
    wgpuSurfaceRelease(surface->surface);
    delete surface;
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceSupportKHR(
        VkPhysicalDevice, uint32_t, VkSurfaceKHR, VkBool32* pSupported) {
    *pSupported = VK_TRUE;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        VkPhysicalDevice, VkSurfaceKHR surface, VkSurfaceCapabilitiesKHR* caps) {
    *caps = {};
    uint32_t w, h;
    canvasSize(surface->canvas, w, h);
    caps->minImageCount = 2;
    caps->maxImageCount = 3;
    caps->currentExtent = {w, h};
    caps->minImageExtent = {1, 1};
    caps->maxImageExtent = {gpu().limits.maxTextureDimension2D, gpu().limits.maxTextureDimension2D};
    caps->maxImageArrayLayers = 1;
    caps->supportedTransforms = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    caps->currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    caps->supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    caps->supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfaceFormatsKHR(
        VkPhysicalDevice, VkSurfaceKHR, uint32_t* pCount, VkSurfaceFormatKHR* pFormats) {
    // A canvas is bgra8unorm or rgba8unorm, never sRGB - but it can be drawn
    // to through an sRGB view of itself, so that is offered beside it.
    //
    // Only the browser's preferred one is offered. A renderer that asks for
    // the other gets it - vk-bootstrap takes its desired format wherever it
    // appears - and the browser then copies the whole canvas every frame to
    // convert it.
    const bool preferRgba = EM_ASM_INT({
        return navigator.gpu.getPreferredCanvasFormat() === 'rgba8unorm' ? 1 : 0;
    }) != 0;
    const VkFormat base = preferRgba ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_B8G8R8A8_UNORM;
    const VkFormat srgb = preferRgba ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_B8G8R8A8_SRGB;
    const VkSurfaceFormatKHR formats[] = {
        {base, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
        {srgb, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR},
    };
    return fillArray(pCount, pFormats, formats, 2);
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPhysicalDeviceSurfacePresentModesKHR(
        VkPhysicalDevice, VkSurfaceKHR, uint32_t* pCount, VkPresentModeKHR* pModes) {
    // The browser presents once per display refresh, and that is all.
    static const VkPresentModeKHR modes[] = {VK_PRESENT_MODE_FIFO_KHR};
    return fillArray(pCount, pModes, modes, 1);
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateSwapchainKHR(
        VkDevice, const VkSwapchainCreateInfoKHR* ci, const VkAllocationCallbacks*,
        VkSwapchainKHR* pSwapchain) {
    auto* sc = new VkSwapchainKHR_T();
    sc->surface = ci->surface;
    sc->format = ci->imageFormat;
    sc->extent = ci->imageExtent;

    const bool rgba = ci->imageFormat == VK_FORMAT_R8G8B8A8_UNORM ||
                      ci->imageFormat == VK_FORMAT_R8G8B8A8_SRGB;
    const bool srgb = ci->imageFormat == VK_FORMAT_B8G8R8A8_SRGB ||
                      ci->imageFormat == VK_FORMAT_R8G8B8A8_SRGB;
    WGPUTextureFormat base = rgba ? WGPUTextureFormat_RGBA8Unorm : WGPUTextureFormat_BGRA8Unorm;
    WGPUTextureFormat view = rgba ? WGPUTextureFormat_RGBA8UnormSrgb : WGPUTextureFormat_BGRA8UnormSrgb;

    // ?offscreen (WOWEE_OFFSCREEN): the swapchain images are ordinary
    // textures and the canvas is never touched. For automated runs in a
    // headless browser, whose canvases lose the device on first draw; frame
    // capture reads the presented image either way.
    static const bool offscreen = std::getenv("WOWEE_OFFSCREEN") != nullptr;
    const uint32_t imageCount = std::clamp(ci->minImageCount, 2u, 3u);
    if (offscreen) {
        for (uint32_t i = 0; i < imageCount; ++i) {
            auto* img = new VkImage_T();
            img->format = ci->imageFormat;
            img->extent = {ci->imageExtent.width, ci->imageExtent.height, 1};
            img->usage = ci->imageUsage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            sc->images.push_back(img);
        }
        sc->offscreen = true;
        *pSwapchain = sc;
        return VK_SUCCESS;
    }

    WGPUSurfaceConfiguration config = WGPU_SURFACE_CONFIGURATION_INIT;
    config.device = gpu().device;
    config.format = base;
    // Drawn to, and copied from for frame capture. Asking for more than the
    // canvas supports loses the device outright in some browsers - headless
    // Chromium's cannot back a canvas that is also sampled or copied into.
    config.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc;
    config.width = ci->imageExtent.width;
    config.height = ci->imageExtent.height;
    config.alphaMode = WGPUCompositeAlphaMode_Opaque;
    config.presentMode = WGPUPresentMode_Fifo;
    if (srgb) {
        config.viewFormatCount = 1;
        config.viewFormats = &view;
    }
    wgpuSurfaceConfigure(sc->surface->surface, &config);

    for (uint32_t i = 0; i < imageCount; ++i) {
        auto* img = new VkImage_T();
        img->format = ci->imageFormat;
        img->extent = {ci->imageExtent.width, ci->imageExtent.height, 1};
        img->usage = ci->imageUsage;
        img->swapchain = sc;
        sc->images.push_back(img);
    }
    *pSwapchain = sc;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroySwapchainKHR(VkDevice, VkSwapchainKHR sc, const VkAllocationCallbacks*) {
    if (!sc) return;
    if (sc->current) wgpuTextureRelease(sc->current);
    for (auto* img : sc->images) {
        if (img->gpu) {
            wgpuTextureDestroy(img->gpu);
            wgpuTextureRelease(img->gpu);
        }
        delete img;
    }
    delete sc;
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetSwapchainImagesKHR(
        VkDevice, VkSwapchainKHR sc, uint32_t* pCount, VkImage* pImages) {
    std::vector<VkImage> images(sc->images.begin(), sc->images.end());
    return fillArray(pCount, pImages, images.data(), static_cast<uint32_t>(images.size()));
}

VKAPI_ATTR VkResult VKAPI_CALL vkAcquireNextImageKHR(
        VkDevice, VkSwapchainKHR sc, uint64_t, VkSemaphore semaphore, VkFence fence,
        uint32_t* pImageIndex) {
    *pImageIndex = sc->next;
    sc->next = (sc->next + 1) % static_cast<uint32_t>(sc->images.size());
    if (semaphore) semaphore->value.fetch_add(1);
    if (fence) fence->signalled = true;

    uint32_t w, h;
    canvasSize(sc->surface->canvas, w, h);
    if (w != sc->extent.width || h != sc->extent.height) return VK_SUBOPTIMAL_KHR;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkQueuePresentKHR(VkQueue, const VkPresentInfoKHR* info) {
    // The browser shows the canvas texture when this frame's callback
    // returns; presenting is only letting go of it.
    for (uint32_t i = 0; i < info->swapchainCount; ++i) {
        VkSwapchainKHR sc = info->pSwapchains[i];
        const bool bgra = sc->format == VK_FORMAT_B8G8R8A8_UNORM || sc->format == VK_FORMAT_B8G8R8A8_SRGB;
        WGPUTexture shown = sc->offscreen ? sc->images[info->pImageIndices[i]]->gpu : sc->current;
        maybeCapture(shown, sc->extent.width, sc->extent.height, bgra);
        if (sc->current) {
            wgpuTextureRelease(sc->current);
            sc->current = nullptr;
        }
        ++sc->frame;
        if (info->pResults) info->pResults[i] = VK_SUCCESS;
    }
    return VK_SUCCESS;
}

// Synchronisation ---------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL vkCreateFence(
        VkDevice, const VkFenceCreateInfo* ci, const VkAllocationCallbacks*, VkFence* pFence) {
    auto* f = new VkFence_T();
    f->signalled = (ci->flags & VK_FENCE_CREATE_SIGNALED_BIT) != 0;
    *pFence = f;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyFence(VkDevice, VkFence fence, const VkAllocationCallbacks*) {
    delete fence;
}

VKAPI_ATTR VkResult VKAPI_CALL vkResetFences(VkDevice, uint32_t count, const VkFence* fences) {
    for (uint32_t i = 0; i < count; ++i) fences[i]->signalled = false;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetFenceStatus(VkDevice, VkFence fence) {
    if (onMainThread()) drainMainQueue();
    return fence->signalled ? VK_SUCCESS : VK_NOT_READY;
}

VKAPI_ATTR VkResult VKAPI_CALL vkWaitForFences(
        VkDevice, uint32_t count, const VkFence* fences, VkBool32 waitAll, uint64_t timeout) {
    auto done = [&] {
        uint32_t signalled = 0;
        for (uint32_t i = 0; i < count; ++i) signalled += fences[i]->signalled ? 1 : 0;
        return waitAll ? signalled == count : signalled > 0;
    };
    if (onMainThread()) {
        // The main thread is where fences are signalled; one it is waiting on
        // is either signalled once its queued submit runs, or never will be.
        drainMainQueue();
        return done() ? VK_SUCCESS : VK_TIMEOUT;
    }
    // A worker waits for the main thread to run what it submitted.
    const auto deadline = std::chrono::steady_clock::now() +
        std::chrono::nanoseconds(std::min<uint64_t>(timeout, 60ull * 1000000000ull));
    while (!done()) {
        if (std::chrono::steady_clock::now() >= deadline) return VK_TIMEOUT;
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateSemaphore(
        VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*, VkSemaphore* pSemaphore) {
    *pSemaphore = new VkSemaphore_T();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroySemaphore(VkDevice, VkSemaphore s, const VkAllocationCallbacks*) {
    delete s;
}

VKAPI_ATTR VkResult VKAPI_CALL vkWaitSemaphores(VkDevice, const VkSemaphoreWaitInfo*, uint64_t) {
    // Only reached with timeline semaphores, which are not offered.
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateQueryPool(
        VkDevice, const VkQueryPoolCreateInfo* ci, const VkAllocationCallbacks*, VkQueryPool* pPool) {
    auto* q = new VkQueryPool_T();
    q->count = ci->queryCount;
    *pPool = q;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyQueryPool(VkDevice, VkQueryPool pool, const VkAllocationCallbacks*) {
    delete pool;
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetQueryPoolResults(
        VkDevice, VkQueryPool, uint32_t, uint32_t, size_t, void*, VkDeviceSize, VkQueryResultFlags) {
    // Timestamps are not offered (timestampValidBits is 0), so nothing that
    // checks for them asks; anything that does gets "not yet".
    return VK_NOT_READY;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineCache(
        VkDevice, const VkPipelineCacheCreateInfo*, const VkAllocationCallbacks*, VkPipelineCache* p) {
    *p = new VkPipelineCache_T();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineCache(VkDevice, VkPipelineCache c, const VkAllocationCallbacks*) {
    delete c;
}

VKAPI_ATTR VkResult VKAPI_CALL vkGetPipelineCacheData(VkDevice, VkPipelineCache, size_t* pSize, void*) {
    *pSize = 0;
    return VK_SUCCESS;
}

} // extern "C"
