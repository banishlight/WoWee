// Memory, buffers, images, views, samplers and shader modules.
//
// VMA runs unchanged on top of this: it allocates VkDeviceMemory blocks and
// binds buffers into them at offsets. A block of host-visible memory is a CPU
// array, so a buffer's mapped pointer is just the block's array plus the
// buffer's offset, and the GPU copy of the buffer is brought up to date from
// it when a submission uses it (see replay.cpp).

#include "wgpu_layer.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

#include "core/logger.hpp"

using namespace wgpuvk;

namespace {

constexpr VkDeviceSize kBufferAlignment = 256;

VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize a) { return (v + a - 1) / a * a; }

WGPUStringView sv(const std::string& s) { return {s.c_str(), s.size()}; }

WGPUAddressMode addressMode(VkSamplerAddressMode m) {
    switch (m) {
    case VK_SAMPLER_ADDRESS_MODE_REPEAT:          return WGPUAddressMode_Repeat;
    case VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT: return WGPUAddressMode_MirrorRepeat;
    default:                                      return WGPUAddressMode_ClampToEdge;
    }
}

WGPUCompareFunction compareFunction(VkCompareOp op) {
    switch (op) {
    case VK_COMPARE_OP_NEVER:            return WGPUCompareFunction_Never;
    case VK_COMPARE_OP_LESS:             return WGPUCompareFunction_Less;
    case VK_COMPARE_OP_EQUAL:            return WGPUCompareFunction_Equal;
    case VK_COMPARE_OP_LESS_OR_EQUAL:    return WGPUCompareFunction_LessEqual;
    case VK_COMPARE_OP_GREATER:          return WGPUCompareFunction_Greater;
    case VK_COMPARE_OP_NOT_EQUAL:        return WGPUCompareFunction_NotEqual;
    case VK_COMPARE_OP_GREATER_OR_EQUAL: return WGPUCompareFunction_GreaterEqual;
    default:                             return WGPUCompareFunction_Always;
    }
}

// The shader manifest ------------------------------------------------------------

uint64_t fnv1a64(const uint8_t* data, size_t n) {
    uint64_t h = 0xCBF29CE484222325ull;
    for (size_t i = 0; i < n; ++i) {
        h ^= data[i];
        h *= 0x100000001B3ull;
    }
    return h;
}

std::string readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

struct Manifest {
    std::unordered_map<uint64_t, ShaderInfo> shaders;
};

const Manifest& manifest() {
    static const Manifest m = [] {
        Manifest out;
        const std::string dir = "assets/shaders/";
        const std::string text = readFile(dir + "manifest.json");
        if (text.empty()) {
            LOG_ERROR("WebGPU: no shader manifest at ", dir, "manifest.json");
            return out;
        }
        const auto json = nlohmann::json::parse(text, nullptr, false);
        if (json.is_discarded()) {
            LOG_ERROR("WebGPU: shader manifest does not parse");
            return out;
        }
        for (auto it = json.begin(); it != json.end(); ++it) {
            const auto& e = it.value();
            ShaderInfo info;
            info.name = e.value("name", "");
            info.wgsl = readFile(dir + e.value("wgsl", ""));
            const std::string stage = e.value("stage", "");
            info.stage = stage == "vertex"   ? WGPUShaderStage_Vertex
                       : stage == "fragment" ? WGPUShaderStage_Fragment
                                             : WGPUShaderStage_Compute;
            for (const auto& r : e["resources"]) {
                ShaderResource res;
                res.group = r.value("group", 0u);
                res.binding = r.value("binding", 0u);
                const std::string kind = r.value("kind", "");
                using K = ShaderResource::Kind;
                res.kind = kind == "uniform"            ? K::Uniform
                         : kind == "storage"            ? K::Storage
                         : kind == "read-only-storage"  ? K::ReadOnlyStorage
                         : kind == "texture"            ? K::Texture
                         : kind == "sampler"            ? K::Sampler
                         : kind == "comparison-sampler" ? K::ComparisonSampler
                                                        : K::StorageTexture;
                res.type = r.value("type", "");
                info.resources.push_back(res);
            }
            out.shaders[std::stoull(it.key(), nullptr, 16)] = std::move(info);
        }
        LOG_INFO("WebGPU: ", out.shaders.size(), " shaders in the manifest");
        return out;
    }();
    return m;
}

} // namespace

namespace wgpuvk {

const ShaderInfo* findShader(const uint32_t* code, size_t bytes) {
    const uint64_t h = fnv1a64(reinterpret_cast<const uint8_t*>(code), bytes);
    const auto& m = manifest().shaders;
    auto it = m.find(h);
    return it == m.end() ? nullptr : &it->second;
}

} // namespace wgpuvk

// Object methods ---------------------------------------------------------------------

uint8_t* VkDeviceMemory_T::hostPtr() {
    if (!hostVisible) return nullptr;
    if (!shadow) {
        shadow.reset(new uint8_t[size]);
        std::memset(shadow.get(), 0, size);
    }
    return shadow.get();
}

const uint8_t* VkBuffer_T::hostData() const {
    if (!memory || !memory->shadow) return nullptr;
    return memory->shadow.get() + memoryOffset;
}

WGPUBuffer VkBuffer_T::ensureGpu() {
    if (gpu) return gpu;
    WGPUBufferDescriptor desc = WGPU_BUFFER_DESCRIPTOR_INIT;
    // Never nothing: WebGPU refuses a buffer of no size, where Vulkan allows
    // one and the renderer makes them for empty batches.
    desc.size = std::max<VkDeviceSize>(alignUp(size, 4), 4);
    WGPUBufferUsage u = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
    if (usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)   u |= WGPUBufferUsage_Vertex;
    if (usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT)    u |= WGPUBufferUsage_Index;
    if (usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)  u |= WGPUBufferUsage_Uniform;
    if (usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)  u |= WGPUBufferUsage_Storage;
    if (usage & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) u |= WGPUBufferUsage_Indirect;
    desc.usage = u;
    gpu = wgpuDeviceCreateBuffer(wgpuvk::gpu().device, &desc);
    return gpu;
}

void retainImage(VkImage_T* image) {
    if (image && !image->swapchain) image->refs.fetch_add(1, std::memory_order_relaxed);
}

void releaseImage(VkImage_T* image) {
    // Swapchain images belong to the swapchain and are freed with it.
    if (!image || image->swapchain) return;
    if (image->refs.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
    // Last reference gone. Release (never destroy - that would pull the texture
    // out from under GPU work in flight), and defer even that a few frames,
    // past any command recorded before now and not yet replayed.
    releaseLater([image] {
        if (image->gpu) wgpuTextureRelease(image->gpu);
        delete image;
    });
}

void retainView(VkImageView_T* view) {
    if (view) view->refs.fetch_add(1, std::memory_order_relaxed);
}

void releaseView(VkImageView_T* view) {
    if (!view) return;
    if (view->refs.fetch_sub(1, std::memory_order_acq_rel) != 1) return;
    VkImage_T* image = view->image;
    releaseLater([view] {
        if (view->gpu) wgpuTextureViewRelease(view->gpu);
        delete view;
    });
    releaseImage(image);            // the view let go of its image
}

WGPUTexture VkImage_T::ensureGpu() {
    if (swapchain) {
        if (!swapchain->current) {
            WGPUSurfaceTexture st = WGPU_SURFACE_TEXTURE_INIT;
            wgpuSurfaceGetCurrentTexture(swapchain->surface->surface, &st);
            swapchain->current = st.texture;
        }
        return swapchain->current;
    }
    if (gpu) return gpu;
    const FormatInfo fi = formatInfo(format);
    if (fi.wgpu == WGPUTextureFormat_Undefined) {
        LOG_ERROR("WebGPU: image format ", static_cast<int>(format), " has no WebGPU equivalent");
        return nullptr;
    }
    WGPUTextureDescriptor desc = WGPU_TEXTURE_DESCRIPTOR_INIT;
    desc.dimension = type == VK_IMAGE_TYPE_3D ? WGPUTextureDimension_3D
                   : type == VK_IMAGE_TYPE_1D ? WGPUTextureDimension_1D
                                              : WGPUTextureDimension_2D;
    desc.size = {extent.width, extent.height,
                 type == VK_IMAGE_TYPE_3D ? extent.depth : arrayLayers};
    desc.format = fi.wgpu;
    desc.mipLevelCount = mipLevels;
    desc.sampleCount = sampleCount(samples);
    WGPUTextureUsage u = WGPUTextureUsage_None;
    if (usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT))
        u |= WGPUTextureUsage_TextureBinding;
    if (usage & VK_IMAGE_USAGE_STORAGE_BIT) u |= WGPUTextureUsage_StorageBinding;
    if (usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT))
        u |= WGPUTextureUsage_RenderAttachment;
    if (samples == VK_SAMPLE_COUNT_1_BIT) {
        u |= WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst;
        // A blit into a mip is drawn, so a colour image that is blitted into
        // has to be drawable even though Vulkan only calls it a copy target.
        if (fi.blockDim == 1 && !fi.depth && (usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
            u |= WGPUTextureUsage_RenderAttachment;
    } else {
        u |= WGPUTextureUsage_RenderAttachment;
    }
    desc.usage = u;
    // The sRGB twin, so a mutable-format image can be viewed either way.
    WGPUTextureFormat twin = WGPUTextureFormat_Undefined;
    if (flags & VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT) {
        switch (fi.wgpu) {
        case WGPUTextureFormat_RGBA8Unorm:     twin = WGPUTextureFormat_RGBA8UnormSrgb; break;
        case WGPUTextureFormat_RGBA8UnormSrgb: twin = WGPUTextureFormat_RGBA8Unorm; break;
        case WGPUTextureFormat_BGRA8Unorm:     twin = WGPUTextureFormat_BGRA8UnormSrgb; break;
        case WGPUTextureFormat_BGRA8UnormSrgb: twin = WGPUTextureFormat_BGRA8Unorm; break;
        default: break;
        }
    }
    if (twin != WGPUTextureFormat_Undefined) {
        desc.viewFormatCount = 1;
        desc.viewFormats = &twin;
    }
    gpu = wgpuDeviceCreateTexture(wgpuvk::gpu().device, &desc);
    return gpu;
}

WGPUTextureView VkImageView_T::ensureGpu() {
    if (image->swapchain) {
        if (gpu && gpuFrame == image->swapchain->frame) return gpu;
        if (gpu) wgpuTextureViewRelease(gpu);
        gpu = nullptr;
        gpuFrame = image->swapchain->frame;
    } else if (gpu) {
        return gpu;
    }
    WGPUTexture tex = image->ensureGpu();
    if (!tex) return nullptr;
    const FormatInfo fi = formatInfo(format);
    WGPUTextureViewDescriptor desc = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    desc.format = fi.wgpu;
    switch (viewType) {
    case VK_IMAGE_VIEW_TYPE_1D:         desc.dimension = WGPUTextureViewDimension_1D; break;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY:   desc.dimension = WGPUTextureViewDimension_2DArray; break;
    case VK_IMAGE_VIEW_TYPE_CUBE:       desc.dimension = WGPUTextureViewDimension_Cube; break;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: desc.dimension = WGPUTextureViewDimension_CubeArray; break;
    case VK_IMAGE_VIEW_TYPE_3D:         desc.dimension = WGPUTextureViewDimension_3D; break;
    default:                            desc.dimension = WGPUTextureViewDimension_2D; break;
    }
    desc.baseMipLevel = range.baseMipLevel;
    desc.mipLevelCount = range.levelCount == VK_REMAINING_MIP_LEVELS
        ? image->mipLevels - range.baseMipLevel : range.levelCount;
    desc.baseArrayLayer = range.baseArrayLayer;
    desc.arrayLayerCount = range.layerCount == VK_REMAINING_ARRAY_LAYERS
        ? image->arrayLayers - range.baseArrayLayer : range.layerCount;
    if (fi.stencil && range.aspectMask == VK_IMAGE_ASPECT_DEPTH_BIT)
        desc.aspect = WGPUTextureAspect_DepthOnly;
    else if (fi.stencil && range.aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT)
        desc.aspect = WGPUTextureAspect_StencilOnly;
    else
        desc.aspect = WGPUTextureAspect_All;
    gpu = wgpuTextureCreateView(tex, &desc);
    return gpu;
}

WGPUSampler VkSampler_T::ensureGpu() {
    if (gpu) return gpu;
    WGPUSamplerDescriptor desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    desc.addressModeU = addressMode(info.addressModeU);
    desc.addressModeV = addressMode(info.addressModeV);
    desc.addressModeW = addressMode(info.addressModeW);
    desc.magFilter = info.magFilter == VK_FILTER_LINEAR ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
    desc.minFilter = info.minFilter == VK_FILTER_LINEAR ? WGPUFilterMode_Linear : WGPUFilterMode_Nearest;
    desc.mipmapFilter = info.mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR
        ? WGPUMipmapFilterMode_Linear : WGPUMipmapFilterMode_Nearest;
    desc.lodMinClamp = std::max(0.0f, info.minLod);
    desc.lodMaxClamp = std::clamp(info.maxLod, desc.lodMinClamp, 32.0f);
    if (info.compareEnable) desc.compare = compareFunction(info.compareOp);
    // Anisotropy is only allowed with every filter linear.
    const bool allLinear = desc.magFilter == WGPUFilterMode_Linear &&
                           desc.minFilter == WGPUFilterMode_Linear &&
                           desc.mipmapFilter == WGPUMipmapFilterMode_Linear;
    desc.maxAnisotropy = (info.anisotropyEnable && allLinear)
        ? static_cast<uint16_t>(std::clamp(info.maxAnisotropy, 1.0f, 16.0f)) : 1;
    gpu = wgpuDeviceCreateSampler(wgpuvk::gpu().device, &desc);
    return gpu;
}

WGPUSampler VkSampler_T::ensureNearest() {
    if (nearest) return nearest;
    WGPUSamplerDescriptor desc = WGPU_SAMPLER_DESCRIPTOR_INIT;
    desc.addressModeU = addressMode(info.addressModeU);
    desc.addressModeV = addressMode(info.addressModeV);
    desc.addressModeW = addressMode(info.addressModeW);
    desc.lodMinClamp = std::max(0.0f, info.minLod);
    desc.lodMaxClamp = std::clamp(info.maxLod, desc.lodMinClamp, 32.0f);
    nearest = wgpuDeviceCreateSampler(wgpuvk::gpu().device, &desc);
    return nearest;
}

WGPUShaderModule VkShaderModule_T::ensureGpu() {
    if (gpu || !shader) return gpu;
    WGPUShaderSourceWGSL src = WGPU_SHADER_SOURCE_WGSL_INIT;
    src.code = sv(shader->wgsl);
    WGPUShaderModuleDescriptor desc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    desc.nextInChain = &src.chain;
    desc.label = sv(shader->name);
    // Compile errors are only reported here; every pipeline built from the
    // module after this just says it is invalid, without saying why.
    wgpuDevicePushErrorScope(wgpuvk::gpu().device, WGPUErrorFilter_Validation);
    gpu = wgpuDeviceCreateShaderModule(wgpuvk::gpu().device, &desc);
    WGPUPopErrorScopeCallbackInfo cb = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = [](WGPUPopErrorScopeStatus, WGPUErrorType type, WGPUStringView message,
                     void* userdata, void*) {
        if (type != WGPUErrorType_NoError)
            LOG_ERROR("WebGPU shader ", static_cast<const ShaderInfo*>(userdata)->name,
                      " does not compile: ", std::string(message.data, message.length));
    };
    cb.userdata1 = const_cast<ShaderInfo*>(shader);
    wgpuDevicePopErrorScope(wgpuvk::gpu().device, cb);
    return gpu;
}

extern "C" {

// Memory ---------------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL vkAllocateMemory(
        VkDevice, const VkMemoryAllocateInfo* ai, const VkAllocationCallbacks*, VkDeviceMemory* pMemory) {
    auto* m = new VkDeviceMemory_T();
    m->size = ai->allocationSize;
    m->typeIndex = ai->memoryTypeIndex;
    m->hostVisible = ai->memoryTypeIndex != kMemDeviceLocal;
    *pMemory = m;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkFreeMemory(VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks*) {
    if (memory) releaseLater([memory] { delete memory; });
}

VKAPI_ATTR VkResult VKAPI_CALL vkMapMemory(
        VkDevice, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize, VkMemoryMapFlags, void** ppData) {
    uint8_t* p = memory->hostPtr();
    if (!p) return VK_ERROR_MEMORY_MAP_FAILED;
    *ppData = p + offset;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkUnmapMemory(VkDevice, VkDeviceMemory) {}

VKAPI_ATTR VkResult VKAPI_CALL vkFlushMappedMemoryRanges(
        VkDevice, uint32_t count, const VkMappedMemoryRange* ranges) {
    for (uint32_t i = 0; i < count; ++i) {
        VkDeviceMemory_T* m = ranges[i].memory;
        const VkDeviceSize lo = ranges[i].offset;
        const VkDeviceSize hi = ranges[i].size == VK_WHOLE_SIZE ? m->size : lo + ranges[i].size;
        std::lock_guard<std::mutex> lock(m->mutex);
        for (VkBuffer_T* b : m->buffers) {
            const VkDeviceSize bLo = std::max(lo, b->memoryOffset);
            const VkDeviceSize bHi = std::min(hi, b->memoryOffset + b->size);
            if (bLo >= bHi) continue;
            b->dirty.emplace_back(bLo - b->memoryOffset, bHi - b->memoryOffset);
            b->everFlushed = true;
        }
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkInvalidateMappedMemoryRanges(VkDevice, uint32_t, const VkMappedMemoryRange*) {
    return VK_SUCCESS;
}

// Buffers ---------------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL vkCreateBuffer(
        VkDevice, const VkBufferCreateInfo* ci, const VkAllocationCallbacks*, VkBuffer* pBuffer) {
    auto* b = new VkBuffer_T();
    b->size = ci->size;
    b->usage = ci->usage;
    *pBuffer = b;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyBuffer(VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
    if (!buffer) return;
    if (VkDeviceMemory_T* m = buffer->memory) {
        std::lock_guard<std::mutex> lock(m->mutex);
        m->buffers.erase(std::remove(m->buffers.begin(), m->buffers.end(), buffer), m->buffers.end());
    }
    releaseLater([buffer] {
        // Release, never destroy: the GPU may still be reading this buffer
        // from a submission in flight - our fence signals when a frame is
        // recorded, not when the GPU has run it, so the game frees resources
        // the queue still needs. Release lets WebGPU keep the buffer alive
        // until that work is done and free it then; destroy pulls it out from
        // under the queue ("Destroyed buffer used in a submit").
        if (buffer->gpu) wgpuBufferRelease(buffer->gpu);
        delete buffer;
    });
}

static void bufferRequirements(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryRequirements& r) {
    r.size = alignUp(size, 4);
    r.alignment = kBufferAlignment;
    // A large buffer the GPU reads while drawing may not have coherent
    // memory: uploaded whole, it would cost its full size at every submission
    // (see kWholeUploadMax). Staging buffers - copy sources only - never are
    // uploaded that way, and VMA's CPU_ONLY usage requires coherent memory.
    constexpr VkBufferUsageFlags drawn = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    r.memoryTypeBits = (size > kWholeUploadMax && (usage & drawn))
        ? (1u << kMemDeviceLocal) | (1u << kMemHostNonCoherent)
        : (1u << kMemDeviceLocal) | (1u << kMemHostCoherent) | (1u << kMemHostNonCoherent);
}

VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements(
        VkDevice, VkBuffer buffer, VkMemoryRequirements* r) {
    bufferRequirements(buffer->size, buffer->usage, *r);
}

VKAPI_ATTR void VKAPI_CALL vkGetBufferMemoryRequirements2(
        VkDevice, const VkBufferMemoryRequirementsInfo2* info, VkMemoryRequirements2* r) {
    bufferRequirements(info->buffer->size, info->buffer->usage, r->memoryRequirements);
    for (auto* h = static_cast<VkBaseOutStructure*>(r->pNext); h; h = h->pNext) {
        if (h->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS) {
            auto* d = reinterpret_cast<VkMemoryDedicatedRequirements*>(h);
            d->prefersDedicatedAllocation = VK_FALSE;
            d->requiresDedicatedAllocation = VK_FALSE;
        }
    }
}

VKAPI_ATTR void VKAPI_CALL vkGetDeviceBufferMemoryRequirements(
        VkDevice, const VkDeviceBufferMemoryRequirements* info, VkMemoryRequirements2* r) {
    bufferRequirements(info->pCreateInfo->size, info->pCreateInfo->usage, r->memoryRequirements);
}

VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory(
        VkDevice, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset) {
    buffer->memory = memory;
    buffer->memoryOffset = offset;
    std::lock_guard<std::mutex> lock(memory->mutex);
    memory->buffers.push_back(buffer);
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkBindBufferMemory2(
        VkDevice device, uint32_t count, const VkBindBufferMemoryInfo* infos) {
    for (uint32_t i = 0; i < count; ++i)
        vkBindBufferMemory(device, infos[i].buffer, infos[i].memory, infos[i].memoryOffset);
    return VK_SUCCESS;
}

// Images ----------------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL vkCreateImage(
        VkDevice, const VkImageCreateInfo* ci, const VkAllocationCallbacks*, VkImage* pImage) {
    if (formatInfo(ci->format).wgpu == WGPUTextureFormat_Undefined) {
        LOG_ERROR("WebGPU: vkCreateImage with unsupported format ", static_cast<int>(ci->format));
        return VK_ERROR_FORMAT_NOT_SUPPORTED;
    }
    auto* img = new VkImage_T();
    img->type = ci->imageType;
    img->format = ci->format;
    img->extent = ci->extent;
    img->mipLevels = ci->mipLevels;
    img->arrayLayers = ci->arrayLayers;
    img->samples = ci->samples;
    img->usage = ci->usage;
    img->flags = ci->flags;
    *pImage = img;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyImage(VkDevice, VkImage image, const VkAllocationCallbacks*) {
    if (!image || image->swapchain) return;
    releaseImage(image);   // the app's own reference; freed once no view holds it
}

static void imageRequirements(const VkImage_T& img, VkMemoryRequirements& r) {
    // Nothing is ever read through it - images live in WebGPU textures - but
    // VMA sizes its blocks from this, so it should be roughly right.
    const FormatInfo fi = formatInfo(img.format);
    VkDeviceSize total = 0;
    uint32_t w = img.extent.width, h = img.extent.height;
    for (uint32_t m = 0; m < img.mipLevels; ++m) {
        const VkDeviceSize bw = (w + fi.blockDim - 1) / fi.blockDim;
        const VkDeviceSize bh = (h + fi.blockDim - 1) / fi.blockDim;
        total += bw * bh * std::max(fi.blockBytes, 1u);
        w = std::max(1u, w / 2);
        h = std::max(1u, h / 2);
    }
    total *= std::max(img.arrayLayers, img.extent.depth) * static_cast<uint32_t>(img.samples);
    r.size = alignUp(std::max<VkDeviceSize>(total, 4), kBufferAlignment);
    r.alignment = kBufferAlignment;
    r.memoryTypeBits = 0b001;
}

VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements(
        VkDevice, VkImage image, VkMemoryRequirements* r) {
    imageRequirements(*image, *r);
}

VKAPI_ATTR void VKAPI_CALL vkGetImageMemoryRequirements2(
        VkDevice, const VkImageMemoryRequirementsInfo2* info, VkMemoryRequirements2* r) {
    imageRequirements(*info->image, r->memoryRequirements);
    for (auto* h = static_cast<VkBaseOutStructure*>(r->pNext); h; h = h->pNext) {
        if (h->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS) {
            auto* d = reinterpret_cast<VkMemoryDedicatedRequirements*>(h);
            d->prefersDedicatedAllocation = VK_FALSE;
            d->requiresDedicatedAllocation = VK_FALSE;
        }
    }
}

VKAPI_ATTR void VKAPI_CALL vkGetDeviceImageMemoryRequirements(
        VkDevice, const VkDeviceImageMemoryRequirements* info, VkMemoryRequirements2* r) {
    VkImage_T tmp;
    tmp.format = info->pCreateInfo->format;
    tmp.extent = info->pCreateInfo->extent;
    tmp.mipLevels = info->pCreateInfo->mipLevels;
    tmp.arrayLayers = info->pCreateInfo->arrayLayers;
    tmp.samples = info->pCreateInfo->samples;
    imageRequirements(tmp, r->memoryRequirements);
}

VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) {
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkBindImageMemory2(VkDevice, uint32_t, const VkBindImageMemoryInfo*) {
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateImageView(
        VkDevice, const VkImageViewCreateInfo* ci, const VkAllocationCallbacks*, VkImageView* pView) {
    auto* v = new VkImageView_T();
    v->image = ci->image;
    retainImage(v->image);
    v->viewType = ci->viewType;
    v->format = ci->format;
    v->range = ci->subresourceRange;
    *pView = v;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyImageView(VkDevice, VkImageView view, const VkAllocationCallbacks*) {
    if (!view) return;
    releaseView(view);   // the app's own reference; freed once no descriptor holds it
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateSampler(
        VkDevice, const VkSamplerCreateInfo* ci, const VkAllocationCallbacks*, VkSampler* pSampler) {
    auto* s = new VkSampler_T();
    s->info = *ci;
    s->info.pNext = nullptr;
    *pSampler = s;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroySampler(VkDevice, VkSampler sampler, const VkAllocationCallbacks*) {
    if (!sampler) return;
    releaseLater([sampler] {
        if (sampler->gpu) wgpuSamplerRelease(sampler->gpu);
        delete sampler;
    });
}

// Shaders ---------------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL vkCreateShaderModule(
        VkDevice, const VkShaderModuleCreateInfo* ci, const VkAllocationCallbacks*, VkShaderModule* pModule) {
    const ShaderInfo* shader = findShader(ci->pCode, ci->codeSize);
    if (!shader) {
        // A module the staging step never saw - rebuilt with a different glslc,
        // or embedded somewhere it does not look. The pipelines that use it
        // are skipped rather than failing the whole renderer.
        LOG_ERROR("WebGPU: no WGSL translation for a ", ci->codeSize,
                  "-byte SPIR-V module (re-run tools/wasm_shaders.py)");
    }
    auto* m = new VkShaderModule_T();
    m->shader = shader;
    *pModule = m;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyShaderModule(VkDevice, VkShaderModule module, const VkAllocationCallbacks*) {
    // Pipelines keep pointers to their modules and compile lazily, so a module
    // is kept for the life of the device. They are small and few.
    (void)module;
}

} // extern "C"
