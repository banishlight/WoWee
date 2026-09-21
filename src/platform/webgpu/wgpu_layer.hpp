#pragma once

// The Vulkan subset the renderer uses, implemented on WebGPU.
//
// The browser has no Vulkan. Rather than a second renderer, the wasm build
// links this layer in place of a Vulkan loader: every vk* entry point the
// client, vk-bootstrap, VMA and ImGui's Vulkan backend call is defined here,
// and the renderer runs unchanged on top of it.
//
// How it maps:
//
//   Objects      Every handle is a pointer to one of the structs below
//                (VK_USE_64_BIT_PTR_DEFINES is on for this build). WebGPU
//                objects are created lazily, on the main thread, the first
//                time a submission needs them - WebGPU objects are JS objects
//                and exist only on the thread that owns the page, while the
//                client creates resources and records commands on workers.
//
//   Commands     vkCmd* appends to the command buffer's own list, on any
//                thread. vkQueueSubmit replays the lists into a WebGPU command
//                encoder on the main thread (see replay.cpp).
//
//   Memory       Host-visible memory is a CPU-side copy of its contents. What
//                a submission reads from it is uploaded when it is submitted:
//                whole, for buffers small enough to be in coherent memory, and
//                only the flushed ranges for large ones, which are only
//                offered non-coherent memory (kWholeUploadMax). Copies out of
//                it into a buffer or image become queue writes straight from
//                the copy, so a staging buffer never needs a GPU buffer.
//
//   Bindings     WGSL has no combined image sampler, so binding b of a
//                descriptor set is WGSL binding 2b, and the sampler half of a
//                combined image sampler 2b + 1 (tools/wasm_shaders.py does the
//                same to the shaders). Push constants are a uniform buffer at
//                group 0, binding kPushConstantBinding, with a dynamic offset.
//
//   Sync         WebGPU orders a queue's work and tracks hazards itself, so
//                barriers and layout transitions are nothing, semaphores are
//                nothing, and a fence is signalled when its submission is
//                handed to the queue.

#include <vulkan/vulkan.h>
#include <webgpu/webgpu.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace wgpuvk {

constexpr uint32_t kPushConstantBinding = 900;
constexpr uint32_t kMaxPushConstantBytes = 256;
constexpr uint32_t kMaxBoundSets = 4;
constexpr uint32_t kMaxVertexBuffers = 8;

/// Host-visible buffers up to this size are uploaded whole whenever a
/// submission uses them. Larger ones only get non-coherent memory, and only
/// their flushed ranges are uploaded (see VkBuffer_T::dirty).
constexpr VkDeviceSize kWholeUploadMax = 4ull * 1024 * 1024;

/// The memory types vkGetPhysicalDeviceMemoryProperties reports.
constexpr uint32_t kMemDeviceLocal = 0;
constexpr uint32_t kMemHostCoherent = 1;
constexpr uint32_t kMemHostNonCoherent = 2;

/// WebGPU draws with one sample or four. Two and eight become four, the same
/// for images and pipelines so that the two always agree.
constexpr uint32_t sampleCount(VkSampleCountFlagBits s) { return s == VK_SAMPLE_COUNT_1_BIT ? 1 : 4; }

/// Vulkan binding b of a descriptor set, as the WGSL numbers it.
constexpr uint32_t textureBinding(uint32_t vkBinding) { return vkBinding * 2; }
constexpr uint32_t samplerBinding(uint32_t vkBinding) { return vkBinding * 2 + 1; }

/// The single WebGPU device the page asked for before main (see webgpu_pre.js).
struct Gpu {
    WGPUInstance instance = nullptr;
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    WGPULimits limits{};
    bool bc = false;               // texture-compression-bc
    bool float32Filterable = false;
    bool depth32Stencil8 = false;
    // One call for a whole indirect batch instead of one per draw. Chromium
    // only, behind its experimental flag, so every use needs the fallback.
    bool multiDraw = false;
};
Gpu& gpu();
bool onMainThread();

// Formats ----------------------------------------------------------------------

struct FormatInfo {
    WGPUTextureFormat wgpu = WGPUTextureFormat_Undefined;
    uint32_t blockBytes = 0;       // bytes per texel, or per 4x4 block
    uint32_t blockDim = 1;         // 1, or 4 for block-compressed formats
    bool depth = false;
    bool stencil = false;
};
FormatInfo formatInfo(VkFormat format);
WGPUVertexFormat vertexFormat(VkFormat format);

// Shaders ----------------------------------------------------------------------

/// What one WGSL global is, from the manifest tools/wasm_shaders.py writes.
struct ShaderResource {
    enum class Kind { Uniform, Storage, ReadOnlyStorage, Texture, Sampler,
                      ComparisonSampler, StorageTexture };
    uint32_t group = 0;
    uint32_t binding = 0;
    Kind kind = Kind::Uniform;
    std::string type;              // the WGSL type, for textures
};

struct ShaderInfo {
    std::string name;
    std::string wgsl;
    WGPUShaderStage stage = WGPUShaderStage_None;
    std::vector<ShaderResource> resources;
};

/// The translation for a SPIR-V module, or nullptr if none was staged.
const ShaderInfo* findShader(const uint32_t* code, size_t bytes);

} // namespace wgpuvk

// Objects ------------------------------------------------------------------------
//
// In the global namespace because that is where vulkan.h declares the handle
// types as pointers to incomplete structs of these names.

struct VkInstance_T {};

struct VkPhysicalDevice_T {};

struct VkQueue_T {};

struct VkDevice_T {
    VkQueue_T queue;
};

struct VkSurfaceKHR_T {
    WGPUSurface surface = nullptr;
    std::string canvas;
};

struct VkImage_T;

struct VkSwapchainKHR_T {
    VkSurfaceKHR_T* surface = nullptr;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent2D extent{};
    std::vector<VkImage_T*> images;
    uint32_t next = 0;
    // The canvas texture for the frame being drawn, fetched on first use and
    // dropped at present: the browser shows it when the frame's task returns.
    WGPUTexture current = nullptr;
    uint64_t frame = 1;
    bool offscreen = false;                  // images are plain textures (see vkCreateSwapchainKHR)
};

struct VkBuffer_T;

struct VkDeviceMemory_T {
    VkDeviceSize size = 0;
    uint32_t typeIndex = 0;
    bool hostVisible = false;
    // The buffers bound into this memory, so a flush of a memory range can
    // be charged to the buffers it covers. Guarded by mutex, as is every
    // bound buffer's dirty list.
    std::mutex mutex;
    std::vector<VkBuffer_T*> buffers;
    // Host-visible contents, made on first map. Never mapped means nothing
    // was ever written from the CPU, and there is nothing to upload.
    std::unique_ptr<uint8_t[]> shadow;
    uint8_t* hostPtr();
};

struct VkBuffer_T {
    VkDeviceSize size = 0;
    VkBufferUsageFlags usage = 0;
    VkDeviceMemory_T* memory = nullptr;
    VkDeviceSize memoryOffset = 0;
    WGPUBuffer gpu = nullptr;
    // Buffer-relative [begin, end) ranges flushed since they were last
    // uploaded - how a large buffer's CPU writes reach the GPU.
    std::vector<std::pair<VkDeviceSize, VkDeviceSize>> dirty;
    bool everFlushed = false;
    bool warnedUnflushed = false;
    // What was last sent to the GPU, for a buffer small enough to be sent
    // whole: most of them hold the same bytes frame after frame, and the
    // comparison is far cheaper than the upload. Empty until the first one.
    std::vector<uint8_t> uploaded;

    WGPUBuffer ensureGpu();
    /// The CPU copy of this buffer's contents, if it lives in mapped memory.
    const uint8_t* hostData() const;
};

struct VkImage_T {
    VkImageType type = VK_IMAGE_TYPE_2D;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent3D extent{};
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    VkImageUsageFlags usage = 0;
    VkImageCreateFlags flags = 0;
    WGPUTexture gpu = nullptr;
    VkSwapchainKHR_T* swapchain = nullptr;   // set for swapchain images
    std::atomic<int> refs{1};                // live views + the app's handle; see releaseImage

    WGPUTexture ensureGpu();
};

struct VkImageView_T {
    VkImage_T* image = nullptr;
    VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkImageSubresourceRange range{};
    WGPUTextureView gpu = nullptr;
    uint64_t gpuFrame = 0;                   // swapchain frame the view is of
    std::atomic<int> refs{1};                // descriptors that name it + the app's handle; see releaseView

    WGPUTextureView ensureGpu();
};

// Views and images are reference counted, so the game can destroy one while a
// descriptor set still names it and a command recorded before the destroy but
// not yet replayed still draws with it - which it does on every zone change. A
// view holds a count on its image, a descriptor holds one on its view (via
// ViewRef); each is torn down only once its last reference goes, the way a
// native driver keeps a resource alive until its work is done.
void retainImage(VkImage_T* image);
void releaseImage(VkImage_T* image);
void retainView(VkImageView_T* view);
void releaseView(VkImageView_T* view);

/// A counted handle to a view, so a descriptor keeps the view alive through
/// every copy the descriptor arrays make of it.
struct ViewRef {
    VkImageView_T* p = nullptr;
    ViewRef() = default;
    ViewRef(VkImageView_T* v) : p(v) { retainView(p); }
    ViewRef(const ViewRef& o) : p(o.p) { retainView(p); }
    ViewRef(ViewRef&& o) noexcept : p(o.p) { o.p = nullptr; }
    ViewRef& operator=(VkImageView_T* v) { retainView(v); releaseView(p); p = v; return *this; }
    ViewRef& operator=(const ViewRef& o) { retainView(o.p); releaseView(p); p = o.p; return *this; }
    ViewRef& operator=(ViewRef&& o) noexcept { if (this != &o) { releaseView(p); p = o.p; o.p = nullptr; } return *this; }
    ~ViewRef() { releaseView(p); }
    VkImageView_T* operator->() const { return p; }
    operator VkImageView_T*() const { return p; }
};

struct VkSampler_T {
    VkSamplerCreateInfo info{};
    WGPUSampler gpu = nullptr;
    WGPUSampler ensureGpu();
    // The same sampler without filtering, for a texture WebGPU will not filter
    // - a depth buffer read as an ordinary texture, say.
    WGPUSampler nearest = nullptr;
    WGPUSampler ensureNearest();
};

struct VkShaderModule_T {
    const wgpuvk::ShaderInfo* shader = nullptr;
    WGPUShaderModule gpu = nullptr;
    WGPUShaderModule ensureGpu();
};

struct VkDescriptorSetLayout_T {
    struct Binding {
        uint32_t binding = 0;
        VkDescriptorType type = VK_DESCRIPTOR_TYPE_MAX_ENUM;
        uint32_t count = 0;
        std::vector<VkSampler_T*> immutableSamplers;
    };
    std::vector<Binding> bindings;
    const Binding* find(uint32_t binding) const;
};

struct VkDescriptorPool_T;

struct VkDescriptorSet_T {
    struct Descriptor {
        VkBuffer_T* buffer = nullptr;
        VkDeviceSize offset = 0;
        VkDeviceSize range = 0;
        ViewRef view;
        VkSampler_T* sampler = nullptr;
    };
    VkDescriptorSetLayout_T* layout = nullptr;
    VkDescriptorPool_T* pool = nullptr;
    // binding -> its descriptors (arrays hold count of them)
    std::unordered_map<uint32_t, std::vector<Descriptor>> descriptors;
    // Bumped by every update, so a bind group built from older contents is
    // known to be stale.
    uint64_t version = 1;

    struct CachedGroup {
        WGPUBindGroupLayout layout;
        uint64_t version;
        uint64_t swapFrame;
        WGPUBindGroup group;
    };
    std::vector<CachedGroup> groups;
    ~VkDescriptorSet_T();
};

struct VkDescriptorPool_T {
    std::vector<VkDescriptorSet_T*> sets;
};

struct VkPipelineLayout_T {
    std::vector<VkDescriptorSetLayout_T*> setLayouts;
    uint32_t pushConstantBytes = 0;
};

struct VkRenderPass_T {
    std::vector<VkAttachmentDescription> attachments;
    std::vector<uint32_t> colors;            // attachment index per color slot
    std::vector<uint32_t> resolves;          // VK_ATTACHMENT_UNUSED if none
    uint32_t depth = VK_ATTACHMENT_UNUSED;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
};

struct VkFramebuffer_T {
    std::vector<VkImageView_T*> attachments;
    uint32_t width = 0, height = 0;
};

struct VkPipeline_T {
    bool compute = false;
    VkPipelineLayout_T* layout = nullptr;

    // Graphics state, kept to build the WebGPU pipeline when it is first drawn
    // with - and again for each depth bias, which WebGPU fixes at creation.
    VkShaderModule_T* vertex = nullptr;
    VkShaderModule_T* fragment = nullptr;
    std::string vertexEntry = "main", fragmentEntry = "main";
    std::vector<VkVertexInputBindingDescription> vertexBindings;
    std::vector<VkVertexInputAttributeDescription> vertexAttributes;
    VkPrimitiveTopology topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkCullModeFlags cullMode = VK_CULL_MODE_NONE;
    VkFrontFace frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    bool depthBiasEnable = false;
    float depthBiasConstant = 0, depthBiasSlope = 0, depthBiasClamp = 0;
    bool dynamicDepthBias = false;
    bool depthClamp = false;
    bool alphaToCoverage = false;
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    bool hasDepthStencil = false;
    std::vector<VkPipelineColorBlendAttachmentState> blends;
    VkRenderPass_T* renderPass = nullptr;
    // Set when the pipeline fixes them rather than taking them per draw.
    bool staticViewport = false, staticScissor = false;
    VkViewport viewport{};
    VkRect2D scissor{};

    // Compute
    VkShaderModule_T* compute_ = nullptr;
    std::string computeEntry = "main";

    WGPUPipelineLayout gpuLayout = nullptr;
    std::array<WGPUBindGroupLayout, wgpuvk::kMaxBoundSets> groupLayouts{};
    std::array<std::vector<WGPUBindGroupLayoutEntry>, wgpuvk::kMaxBoundSets> groupEntries;
    uint32_t groupCount = 0;
    bool pushConstants = false;              // group 0 carries kPushConstantBinding
    // Per group, the bindings that take a dynamic offset, in binding order.
    std::array<std::vector<uint32_t>, wgpuvk::kMaxBoundSets> dynamicBindings;

    struct Variant {
        float constant, slope, clamp;
        WGPURenderPipeline pipeline;
        // Set when WebGPU reports the pipeline invalid (see pipeline.cpp).
        // Drawing with it would invalidate the whole command buffer, so
        // those draws are skipped instead.
        std::shared_ptr<std::atomic<bool>> refused;
    };
    std::vector<Variant> variants;
    WGPUComputePipeline gpuCompute = nullptr;
    std::shared_ptr<std::atomic<bool>> computeRefused = std::make_shared<std::atomic<bool>>(false);
    bool failed = false;

    // (group, Vulkan binding) of each texture bound at the first draw that
    // cannot be filtered. Filled by the replay before the layout is built:
    // WebGPU fixes a texture's sample type in the layout, and only what is
    // actually bound says whether a sampler2D is reading a depth buffer.
    std::vector<std::pair<uint32_t, uint32_t>> unfilterable;
    bool ensureLayout();
    WGPURenderPipeline renderPipeline(float biasConstant, float biasSlope, float biasClamp);
    WGPUComputePipeline computePipeline();
};

struct VkFence_T {
    std::atomic<bool> signalled{false};
};

struct VkSemaphore_T {
    std::atomic<uint64_t> value{0};
};

struct VkQueryPool_T {
    uint32_t count = 0;
};

struct VkPipelineCache_T {};

struct VkCommandPool_T;

// Recorded commands ------------------------------------------------------------

namespace wgpuvk::cmd {

struct BeginRenderPass {
    VkRenderPass_T* renderPass;
    VkFramebuffer_T* framebuffer;
    VkRect2D area;
    std::vector<VkClearValue> clears;
};
struct EndRenderPass {};
struct BindPipeline { VkPipeline_T* pipeline; };
struct BindDescriptorSets {
    uint32_t firstSet;
    std::vector<VkDescriptorSet_T*> sets;
    std::vector<uint32_t> dynamicOffsets;
};
struct BindVertexBuffers {
    uint32_t first;
    std::vector<VkBuffer_T*> buffers;
    std::vector<VkDeviceSize> offsets;
};
struct BindIndexBuffer { VkBuffer_T* buffer; VkDeviceSize offset; VkIndexType type; };
struct PushConstants { uint32_t offset; std::vector<uint8_t> data; };
struct SetViewport { VkViewport viewport; };
struct SetScissor { VkRect2D scissor; };
struct SetDepthBias { float constant, clamp, slope; };
struct Draw { uint32_t vertexCount, instanceCount, firstVertex, firstInstance; };
struct DrawIndexed {
    uint32_t indexCount, instanceCount, firstIndex;
    int32_t vertexOffset;
    uint32_t firstInstance;
};
struct DrawIndexedIndirect { VkBuffer_T* buffer; VkDeviceSize offset; uint32_t count, stride; };
struct Dispatch { uint32_t x, y, z; };
struct CopyBuffer { VkBuffer_T* src; VkBuffer_T* dst; std::vector<VkBufferCopy> regions; };
struct CopyBufferToImage {
    VkBuffer_T* src;
    VkImage_T* dst;
    std::vector<VkBufferImageCopy> regions;
};
struct CopyImageToBuffer {
    VkImage_T* src;
    VkBuffer_T* dst;
    std::vector<VkBufferImageCopy> regions;
};
struct CopyImage { VkImage_T* src; VkImage_T* dst; std::vector<VkImageCopy> regions; };
struct BlitImage {
    VkImage_T* src;
    VkImage_T* dst;
    std::vector<VkImageBlit> regions;
    VkFilter filter;
};
struct ClearColorImage {
    VkImage_T* image;
    VkClearColorValue color;
    std::vector<VkImageSubresourceRange> ranges;
};
struct ClearDepthStencilImage {
    VkImage_T* image;
    VkClearDepthStencilValue value;
    std::vector<VkImageSubresourceRange> ranges;
};
struct FillBuffer { VkBuffer_T* buffer; VkDeviceSize offset, size; uint32_t data; };
struct ExecuteCommands { std::vector<VkCommandBuffer_T*> buffers; };

using Command = std::variant<
    BeginRenderPass, EndRenderPass, BindPipeline, BindDescriptorSets,
    BindVertexBuffers, BindIndexBuffer, PushConstants, SetViewport, SetScissor,
    SetDepthBias, Draw, DrawIndexed, DrawIndexedIndirect, Dispatch, CopyBuffer,
    CopyBufferToImage, CopyImageToBuffer, CopyImage, BlitImage, ClearColorImage,
    ClearDepthStencilImage, FillBuffer, ExecuteCommands>;

} // namespace wgpuvk::cmd

struct VkCommandBuffer_T {
    VkCommandPool_T* pool = nullptr;
    std::vector<wgpuvk::cmd::Command> commands;
};

struct VkCommandPool_T {
    std::vector<VkCommandBuffer_T*> buffers;
};

namespace wgpuvk {

/// Runs one vkQueueSubmit's command buffers on the GPU. Main thread only.
void replay(const std::vector<VkCommandBuffer_T*>& buffers);

/// Work handed in off the main thread, run by the next main-thread entry.
void runOnMain(std::function<void()> fn);
void drainMainQueue();

/// Tears a resource down later, whichever thread asks. The teardown is
/// parked and only run a few frames on (see retireFrame): the game destroys
/// textures and buffers while the GPU, and our own replay, may still be using
/// them, so - like a native driver - we hold them until nothing in flight can
/// name them.
void releaseLater(std::function<void()> fn);

/// Advances the deferral by one frame, running whatever is now old enough to
/// free. Called once a frame, at present, on the main thread.
void retireFrame();

/// Debug: logs frame WOWEE_CAPTURE_FRAME's canvas as a PNG (see capture.cpp).
void maybeCapture(WGPUTexture texture, uint32_t width, uint32_t height, bool bgra);

} // namespace wgpuvk
