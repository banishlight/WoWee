// vkQueueSubmit: a submission's recorded commands, run on WebGPU.
//
// Vulkan binds state and then draws; WebGPU does too, but wants every bind
// group of the pipeline in place, a vertex buffer slot per layout, and push
// constants that do not exist. So binds are only remembered here, and a draw
// first brings WebGPU up to date with whatever changed since the last one.
//
// Copies out of host-visible memory are queue writes rather than GPU copies,
// and a queue write lands before whatever is later submitted - so one that
// follows commands already encoded ends the encoder and submits it first, to
// keep the order the command buffer had.

#include "wgpu_layer.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <unordered_set>

#include "core/logger.hpp"

using namespace wgpuvk;
namespace c = wgpuvk::cmd;

namespace {

constexpr uint64_t kRingBytes = 16ull * 1024 * 1024;
constexpr uint32_t kRingSlot = 256;

WGPUStringView sv(const char* s) { return {s, std::strlen(s)}; }

template <class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

/// Things a bind group can fall back to when the set has nothing at a binding
/// the shader reads - WebGPU refuses a bind group with a hole in it.
struct Dummies {
    WGPUBuffer buffer = nullptr;
    WGPUTextureView view2d = nullptr;
    WGPUTextureView depth2d = nullptr;
    WGPUSampler sampler = nullptr;
    WGPUSampler comparison = nullptr;
};

Dummies& dummies() {
    static Dummies d = [] {
        Dummies x;
        WGPUDevice dev = gpu().device;
        WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
        bd.size = 65536;
        bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        x.buffer = wgpuDeviceCreateBuffer(dev, &bd);

        WGPUTextureDescriptor td = WGPU_TEXTURE_DESCRIPTOR_INIT;
        td.size = {1, 1, 1};
        td.format = WGPUTextureFormat_RGBA8Unorm;
        td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
        x.view2d = wgpuTextureCreateView(wgpuDeviceCreateTexture(dev, &td), nullptr);
        td.format = WGPUTextureFormat_Depth32Float;
        td.usage = WGPUTextureUsage_TextureBinding;
        x.depth2d = wgpuTextureCreateView(wgpuDeviceCreateTexture(dev, &td), nullptr);

        WGPUSamplerDescriptor sd = WGPU_SAMPLER_DESCRIPTOR_INIT;
        x.sampler = wgpuDeviceCreateSampler(dev, &sd);
        sd.compare = WGPUCompareFunction_LessEqual;
        x.comparison = wgpuDeviceCreateSampler(dev, &sd);
        return x;
    }();
    return d;
}

/// The uniform buffer push constants (and blit parameters) are written into,
/// one 256-byte slot per change. Rewritten from the start each submission:
/// queue writes and submissions run in order, so the previous submission has
/// read its slots before the next write reaches the buffer.
struct Ring {
    WGPUBuffer buffer = nullptr;
    std::vector<uint8_t> staging;
    uint64_t used = 0;
    uint64_t flushed = 0;

    Ring() {
        WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
        bd.size = kRingBytes;
        bd.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
        buffer = wgpuDeviceCreateBuffer(gpu().device, &bd);
        staging.resize(kRingBytes);
    }
    /// A slot for size bytes, or UINT32_MAX when the ring is full.
    uint32_t push(const void* data, uint32_t size) {
        if (used + kRingSlot > kRingBytes) return UINT32_MAX;
        const uint32_t at = static_cast<uint32_t>(used);
        std::memcpy(staging.data() + at, data, std::min(size, kRingSlot));
        used += kRingSlot;
        return at;
    }
    void flush() {
        if (used > flushed)
            wgpuQueueWriteBuffer(gpu().queue, buffer, flushed, staging.data() + flushed, used - flushed);
        flushed = used;
    }
    void reset() { used = flushed = 0; }
};

Ring& ring() {
    static Ring r;
    return r;
}

// The blit pipeline: a sampled draw over the destination region, which is how
// vkCmdBlitImage (mip generation, mostly) is done without a blit command.
const char* kBlitWgsl = R"(
struct Params { src: vec4<f32> }
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@group(0) @binding(2) var<uniform> p: Params;
struct VO { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
    var q = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    let c = q[i];
    var o: VO;
    o.pos = vec4(c, 0.0, 1.0);
    o.uv = p.src.xy + vec2(c.x * 0.5 + 0.5, 0.5 - c.y * 0.5) * p.src.zw;
    return o;
}
@fragment fn fs(i: VO) -> @location(0) vec4<f32> {
    return textureSampleLevel(t, s, i.uv, 0.0);
}
)";

struct Blitter {
    WGPUShaderModule module = nullptr;
    WGPUBindGroupLayout layout = nullptr;
    WGPUPipelineLayout pipelineLayout = nullptr;
    std::map<WGPUTextureFormat, WGPURenderPipeline> pipelines;
    WGPUSampler linear = nullptr, nearest = nullptr;

    Blitter() {
        WGPUDevice dev = gpu().device;
        WGPUShaderSourceWGSL src = WGPU_SHADER_SOURCE_WGSL_INIT;
        src.code = sv(kBlitWgsl);
        WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
        md.nextInChain = &src.chain;
        module = wgpuDeviceCreateShaderModule(dev, &md);

        WGPUBindGroupLayoutEntry e[3] = {WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
                                         WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
                                         WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT};
        e[0].binding = 0;
        e[0].visibility = WGPUShaderStage_Fragment;
        e[0].texture.sampleType = WGPUTextureSampleType_Float;
        e[0].texture.viewDimension = WGPUTextureViewDimension_2D;
        e[1].binding = 1;
        e[1].visibility = WGPUShaderStage_Fragment;
        e[1].sampler.type = WGPUSamplerBindingType_Filtering;
        e[2].binding = 2;
        e[2].visibility = WGPUShaderStage_Vertex;
        e[2].buffer.type = WGPUBufferBindingType_Uniform;
        e[2].buffer.hasDynamicOffset = true;
        WGPUBindGroupLayoutDescriptor ld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        ld.entryCount = 3;
        ld.entries = e;
        layout = wgpuDeviceCreateBindGroupLayout(dev, &ld);
        WGPUPipelineLayoutDescriptor pd = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        pd.bindGroupLayoutCount = 1;
        pd.bindGroupLayouts = &layout;
        pipelineLayout = wgpuDeviceCreatePipelineLayout(dev, &pd);

        WGPUSamplerDescriptor sd = WGPU_SAMPLER_DESCRIPTOR_INIT;
        sd.magFilter = sd.minFilter = WGPUFilterMode_Linear;
        linear = wgpuDeviceCreateSampler(dev, &sd);
        sd.magFilter = sd.minFilter = WGPUFilterMode_Nearest;
        nearest = wgpuDeviceCreateSampler(dev, &sd);
    }

    WGPURenderPipeline pipeline(WGPUTextureFormat format) {
        auto it = pipelines.find(format);
        if (it != pipelines.end()) return it->second;
        WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
        target.format = format;
        WGPUFragmentState fs = WGPU_FRAGMENT_STATE_INIT;
        fs.module = module;
        fs.entryPoint = sv("fs");
        fs.targetCount = 1;
        fs.targets = &target;
        WGPURenderPipelineDescriptor d = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
        d.layout = pipelineLayout;
        d.vertex.module = module;
        d.vertex.entryPoint = sv("vs");
        d.fragment = &fs;
        WGPURenderPipeline p = wgpuDeviceCreateRenderPipeline(gpu().device, &d);
        pipelines.emplace(format, p);
        return p;
    }
};

Blitter& blitter() {
    static Blitter b;
    return b;
}

/// Bind groups for a group with no descriptor set behind it - push constants
/// alone, or nothing - one per layout, for the life of the device.
std::map<WGPUBindGroupLayout, WGPUBindGroup>& unsetGroups() {
    static std::map<WGPUBindGroupLayout, WGPUBindGroup> groups;
    return groups;
}

WGPUTextureView singleView(VkImage_T* image, uint32_t mip, uint32_t layer, WGPUTextureFormat format) {
    WGPUTextureViewDescriptor d = WGPU_TEXTURE_VIEW_DESCRIPTOR_INIT;
    d.format = format;
    d.dimension = WGPUTextureViewDimension_2D;
    d.baseMipLevel = mip;
    d.mipLevelCount = 1;
    d.baseArrayLayer = layer;
    d.arrayLayerCount = 1;
    return wgpuTextureCreateView(image->ensureGpu(), &d);
}

WGPUTextureAspect copyAspect(VkImageAspectFlags a, const FormatInfo& fi) {
    if (!fi.stencil) return WGPUTextureAspect_All;
    if (a == VK_IMAGE_ASPECT_DEPTH_BIT) return WGPUTextureAspect_DepthOnly;
    if (a == VK_IMAGE_ASPECT_STENCIL_BIT) return WGPUTextureAspect_StencilOnly;
    return WGPUTextureAspect_All;
}

class Replayer {
public:
    void run(const std::vector<VkCommandBuffer_T*>& buffers) {
        uploadHostBuffers(buffers);
        for (auto* cb : buffers) runList(cb->commands);
        endPasses();
        submit();
        for (auto v : transientViews_) wgpuTextureViewRelease(v);
        for (auto g : transientGroups_) wgpuBindGroupRelease(g);
        transientViews_.clear();
        transientGroups_.clear();
        ring().reset();
    }

private:
    WGPUCommandEncoder enc_ = nullptr;
    bool encoded_ = false;
    WGPURenderPassEncoder pass_ = nullptr;
    WGPUComputePassEncoder cpass_ = nullptr;
    uint32_t passWidth_ = 0, passHeight_ = 0;

    VkPipeline_T* pipeline_ = nullptr;
    WGPURenderPipeline boundRender_ = nullptr;
    WGPUComputePipeline boundCompute_ = nullptr;
    std::array<VkDescriptorSet_T*, kMaxBoundSets> sets_{};
    std::array<std::map<uint32_t, uint32_t>, kMaxBoundSets> setDynamic_;
    std::array<WGPUBindGroup, kMaxBoundSets> boundGroups_{};
    std::array<std::vector<uint32_t>, kMaxBoundSets> boundOffsets_;
    struct VB { VkBuffer_T* buffer = nullptr; VkDeviceSize offset = 0; };
    std::array<VB, 16> vertexBuffers_{};
    std::array<VB, 16> boundVertex_{};
    VkBuffer_T* indexBuffer_ = nullptr;
    VkDeviceSize indexOffset_ = 0;
    VkIndexType indexType_ = VK_INDEX_TYPE_UINT16;
    bool indexDirty_ = true;
    std::array<uint8_t, kMaxPushConstantBytes> pushData_{};
    bool pushDirty_ = true;
    uint32_t pushOffset_ = 0;
    float biasConstant_ = 0, biasClamp_ = 0, biasSlope_ = 0;
    bool haveViewport_ = false, haveScissor_ = false;

    std::vector<WGPUTextureView> transientViews_;
    std::vector<WGPUBindGroup> transientGroups_;

    // Encoders ----------------------------------------------------------------

    WGPUCommandEncoder encoder() {
        if (!enc_) enc_ = wgpuDeviceCreateCommandEncoder(gpu().device, nullptr);
        return enc_;
    }

    void endPasses() {
        if (pass_) {
            wgpuRenderPassEncoderEnd(pass_);
            wgpuRenderPassEncoderRelease(pass_);
            pass_ = nullptr;
        }
        if (cpass_) {
            wgpuComputePassEncoderEnd(cpass_);
            wgpuComputePassEncoderRelease(cpass_);
            cpass_ = nullptr;
        }
        resetBindings();
    }

    void resetBindings() {
        boundRender_ = nullptr;
        boundCompute_ = nullptr;
        boundGroups_ = {};
        for (auto& o : boundOffsets_) o.clear();
        boundVertex_ = {};
        indexDirty_ = true;
    }

    void submit() {
        if (!enc_) return;
        ring().flush();
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc_, nullptr);
        wgpuQueueSubmit(gpu().queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc_);
        enc_ = nullptr;
        encoded_ = false;
    }

    /// Ahead of a queue write: whatever is already encoded has to run first.
    void beforeQueueWrite() {
        endPasses();
        if (encoded_) submit();
    }

    // Host-visible memory --------------------------------------------------------

    static void collect(const std::vector<c::Command>& list, std::unordered_set<VkBuffer_T*>& out) {
        auto add = [&](VkBuffer_T* b) { if (b && b->hostData()) out.insert(b); };
        for (const auto& command : list) {
            std::visit(overloaded{
                [&](const c::BindVertexBuffers& x) { for (auto* b : x.buffers) add(b); },
                [&](const c::BindIndexBuffer& x) { add(x.buffer); },
                [&](const c::DrawIndexedIndirect& x) { add(x.buffer); },
                [&](const c::BindDescriptorSets& x) {
                    for (auto* s : x.sets) {
                        if (!s) continue;
                        for (auto& [binding, ds] : s->descriptors)
                            for (auto& d : ds) add(d.buffer);
                    }
                },
                [&](const c::ExecuteCommands& x) {
                    for (auto* cb : x.buffers) collect(cb->commands, out);
                },
                [](const auto&) {},
            }, command);
        }
    }

    /// Brings the GPU copy of every host-visible buffer this submission reads
    /// up to date with what the CPU wrote, as coherent memory would be.
    void uploadHostBuffers(const std::vector<VkCommandBuffer_T*>& buffers) {
        std::unordered_set<VkBuffer_T*> host;
        for (auto* cb : buffers) collect(cb->commands, host);
        for (VkBuffer_T* b : host) {
            const uint8_t* data = b->hostData();
            const VkDeviceSize room = b->memory->size - b->memoryOffset;
            const VkDeviceSize limit = std::min<VkDeviceSize>((b->size + 3) & ~VkDeviceSize(3), room) & ~VkDeviceSize(3);
            if (b->size <= kWholeUploadMax) {
                if (limit) wgpuQueueWriteBuffer(gpu().queue, b->ensureGpu(), 0, data, limit);
                continue;
            }
            // A large buffer: just what was flushed since the last upload,
            // merged so overlapping flushes go as one write.
            std::vector<std::pair<VkDeviceSize, VkDeviceSize>> ranges;
            {
                std::lock_guard<std::mutex> lock(b->memory->mutex);
                ranges.swap(b->dirty);
                if (!b->everFlushed && !b->warnedUnflushed) {
                    b->warnedUnflushed = true;
                    LOG_WARNING("WebGPU: a ", b->size / (1024 * 1024), " MB host-visible buffer is drawn from "
                                "but was never flushed, so the GPU has none of what the CPU wrote to it");
                }
            }
            std::sort(ranges.begin(), ranges.end());
            for (size_t i = 0; i < ranges.size();) {
                VkDeviceSize lo = ranges[i].first & ~VkDeviceSize(3), hi = ranges[i].second;
                for (++i; i < ranges.size() && ranges[i].first <= hi; ++i) hi = std::max(hi, ranges[i].second);
                hi = std::min((hi + 3) & ~VkDeviceSize(3), limit);
                if (hi > lo) wgpuQueueWriteBuffer(gpu().queue, b->ensureGpu(), lo, data + lo, hi - lo);
            }
        }
    }

    // Commands ----------------------------------------------------------------------

    void runList(const std::vector<c::Command>& list) {
        for (const auto& command : list) {
            std::visit(overloaded{
                [&](const c::BeginRenderPass& x) { beginRenderPass(x); },
                [&](const c::EndRenderPass&) { endPasses(); },
                [&](const c::BindPipeline& x) { bindPipeline(x.pipeline); },
                [&](const c::BindDescriptorSets& x) { bindSets(x); },
                [&](const c::BindVertexBuffers& x) {
                    for (size_t i = 0; i < x.buffers.size() && x.first + i < vertexBuffers_.size(); ++i)
                        vertexBuffers_[x.first + i] = {x.buffers[i], x.offsets[i]};
                },
                [&](const c::BindIndexBuffer& x) {
                    indexBuffer_ = x.buffer;
                    indexOffset_ = x.offset;
                    indexType_ = x.type;
                    indexDirty_ = true;
                },
                [&](const c::PushConstants& x) {
                    const size_t n = std::min<size_t>(x.data.size(), kMaxPushConstantBytes - std::min(x.offset, kMaxPushConstantBytes));
                    std::memcpy(pushData_.data() + x.offset, x.data.data(), n);
                    pushDirty_ = true;
                },
                [&](const c::SetViewport& x) { setViewport(x.viewport); },
                [&](const c::SetScissor& x) { setScissor(x.scissor); },
                [&](const c::SetDepthBias& x) {
                    biasConstant_ = x.constant;
                    biasClamp_ = x.clamp;
                    biasSlope_ = x.slope;
                },
                [&](const c::Draw& x) {
                    if (prepareDraw(false))
                        wgpuRenderPassEncoderDraw(pass_, x.vertexCount, x.instanceCount, x.firstVertex, x.firstInstance);
                },
                [&](const c::DrawIndexed& x) {
                    if (prepareDraw(true))
                        wgpuRenderPassEncoderDrawIndexed(pass_, x.indexCount, x.instanceCount, x.firstIndex,
                                                         x.vertexOffset, x.firstInstance);
                },
                [&](const c::DrawIndexedIndirect& x) {
                    if (!prepareDraw(true)) return;
                    WGPUBuffer b = x.buffer->ensureGpu();
                    for (uint32_t i = 0; i < x.count; ++i)
                        wgpuRenderPassEncoderDrawIndexedIndirect(pass_, b, x.offset + uint64_t(i) * x.stride);
                },
                [&](const c::Dispatch& x) {
                    if (prepareDispatch())
                        wgpuComputePassEncoderDispatchWorkgroups(cpass_, x.x, x.y, x.z);
                },
                [&](const c::CopyBuffer& x) { copyBuffer(x); },
                [&](const c::CopyBufferToImage& x) { copyBufferToImage(x); },
                [&](const c::CopyImageToBuffer& x) { copyImageToBuffer(x); },
                [&](const c::CopyImage& x) { copyImage(x); },
                [&](const c::BlitImage& x) { blitImage(x); },
                [&](const c::ClearColorImage& x) { clearColor(x); },
                [&](const c::ClearDepthStencilImage& x) { clearDepth(x); },
                [&](const c::FillBuffer& x) { fillBuffer(x); },
                [&](const c::ExecuteCommands& x) {
                    for (auto* cb : x.buffers) runList(cb->commands);
                },
            }, command);
        }
    }

    // Render passes ------------------------------------------------------------------

    void beginRenderPass(const c::BeginRenderPass& x) {
        endPasses();
        const VkRenderPass_T* rp = x.renderPass;
        const VkFramebuffer_T* fb = x.framebuffer;
        std::vector<WGPURenderPassColorAttachment> colors;
        for (size_t i = 0; i < rp->colors.size(); ++i) {
            WGPURenderPassColorAttachment ca = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
            const uint32_t a = rp->colors[i];
            if (a != VK_ATTACHMENT_UNUSED) {
                const auto& desc = rp->attachments[a];
                ca.view = fb->attachments[a]->ensureGpu();
                ca.loadOp = desc.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD ? WGPULoadOp_Load : WGPULoadOp_Clear;
                ca.storeOp = desc.storeOp == VK_ATTACHMENT_STORE_OP_STORE ? WGPUStoreOp_Store : WGPUStoreOp_Discard;
                if (a < x.clears.size()) {
                    const auto& cv = x.clears[a].color.float32;
                    ca.clearValue = {cv[0], cv[1], cv[2], cv[3]};
                }
                const uint32_t r = rp->resolves[i];
                if (r != VK_ATTACHMENT_UNUSED) ca.resolveTarget = fb->attachments[r]->ensureGpu();
            }
            colors.push_back(ca);
        }
        WGPURenderPassDepthStencilAttachment da = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
        WGPURenderPassDescriptor desc = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
        desc.colorAttachmentCount = colors.size();
        desc.colorAttachments = colors.data();
        if (rp->depth != VK_ATTACHMENT_UNUSED) {
            const auto& ad = rp->attachments[rp->depth];
            const FormatInfo fi = formatInfo(ad.format);
            da.view = fb->attachments[rp->depth]->ensureGpu();
            da.depthLoadOp = ad.loadOp == VK_ATTACHMENT_LOAD_OP_LOAD ? WGPULoadOp_Load : WGPULoadOp_Clear;
            da.depthStoreOp = ad.storeOp == VK_ATTACHMENT_STORE_OP_STORE ? WGPUStoreOp_Store : WGPUStoreOp_Discard;
            da.depthClearValue = rp->depth < x.clears.size() ? x.clears[rp->depth].depthStencil.depth : 1.0f;
            if (fi.stencil) {
                da.stencilLoadOp = ad.stencilLoadOp == VK_ATTACHMENT_LOAD_OP_LOAD ? WGPULoadOp_Load : WGPULoadOp_Clear;
                da.stencilStoreOp = ad.stencilStoreOp == VK_ATTACHMENT_STORE_OP_STORE ? WGPUStoreOp_Store : WGPUStoreOp_Discard;
                da.stencilClearValue = rp->depth < x.clears.size() ? x.clears[rp->depth].depthStencil.stencil : 0;
            }
            desc.depthStencilAttachment = &da;
        }
        pass_ = wgpuCommandEncoderBeginRenderPass(encoder(), &desc);
        encoded_ = true;
        passWidth_ = fb->width;
        passHeight_ = fb->height;
        // Vulkan leaves these undefined at the start of a pass; WebGPU sets
        // them to the whole target, which is the same thing done kindly.
        haveViewport_ = haveScissor_ = false;
    }

    void setViewport(const VkViewport& v) {
        if (!pass_) return;
        const float x = std::clamp(v.x, 0.0f, float(passWidth_));
        const float y = std::clamp(v.y, 0.0f, float(passHeight_));
        const float w = std::clamp(v.width, 0.0f, float(passWidth_) - x);
        const float h = std::clamp(v.height, 0.0f, float(passHeight_) - y);
        wgpuRenderPassEncoderSetViewport(pass_, x, y, w, h, std::clamp(v.minDepth, 0.0f, 1.0f),
                                         std::clamp(v.maxDepth, 0.0f, 1.0f));
        haveViewport_ = true;
    }

    void setScissor(const VkRect2D& s) {
        if (!pass_) return;
        const uint32_t x = std::min<uint32_t>(std::max(s.offset.x, 0), passWidth_);
        const uint32_t y = std::min<uint32_t>(std::max(s.offset.y, 0), passHeight_);
        const uint32_t w = std::min(s.extent.width, passWidth_ - x);
        const uint32_t h = std::min(s.extent.height, passHeight_ - y);
        wgpuRenderPassEncoderSetScissorRect(pass_, x, y, w, h);
        haveScissor_ = true;
    }

    // State ----------------------------------------------------------------------------

    void bindPipeline(VkPipeline_T* p) {
        pipeline_ = p;
        if (!p->compute && pass_) {
            if (p->staticViewport) setViewport(p->viewport);
            if (p->staticScissor) setScissor(p->scissor);
        }
    }

    void bindSets(const c::BindDescriptorSets& x) {
        size_t dyn = 0;
        for (size_t i = 0; i < x.sets.size(); ++i) {
            const uint32_t index = x.firstSet + static_cast<uint32_t>(i);
            if (index >= kMaxBoundSets) break;
            VkDescriptorSet_T* set = x.sets[i];
            sets_[index] = set;
            setDynamic_[index].clear();
            if (!set) continue;
            // Dynamic offsets come in binding order across the sets bound.
            std::vector<const VkDescriptorSetLayout_T::Binding*> ordered;
            for (const auto& b : set->layout->bindings) ordered.push_back(&b);
            std::sort(ordered.begin(), ordered.end(),
                      [](auto* a, auto* b) { return a->binding < b->binding; });
            for (const auto* b : ordered) {
                if (b->type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC &&
                    b->type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC) continue;
                if (dyn < x.dynamicOffsets.size()) setDynamic_[index][b->binding] = x.dynamicOffsets[dyn];
                dyn += b->count;
            }
        }
    }

    WGPUBindGroup makeGroup(VkPipeline_T* p, uint32_t g) {
        VkDescriptorSet_T* set = g < sets_.size() ? sets_[g] : nullptr;
        const WGPUBindGroupLayout layout = p->groupLayouts[g];
        // Anything drawn to a swapchain view has a new view each frame.
        uint64_t swapFrame = 0;
        if (set) {
            for (auto& [b, ds] : set->descriptors)
                for (auto& d : ds)
                    if (d.view && d.view->image->swapchain) swapFrame = d.view->image->swapchain->frame;
            for (auto& cg : set->groups) {
                if (cg.layout == layout && cg.version == set->version && cg.swapFrame == swapFrame)
                    return cg.group;
            }
        } else {
            auto it = unsetGroups().find(layout);
            if (it != unsetGroups().end()) return it->second;
        }

        const Dummies& dm = dummies();
        std::vector<WGPUBindGroupEntry> entries;
        for (const WGPUBindGroupLayoutEntry& le : p->groupEntries[g]) {
            WGPUBindGroupEntry e = WGPU_BIND_GROUP_ENTRY_INIT;
            e.binding = le.binding;
            const VkDescriptorSet_T::Descriptor* d = nullptr;
            if (set && le.binding != kPushConstantBinding) {
                auto it = set->descriptors.find(le.binding / 2);
                if (it != set->descriptors.end() && !it->second.empty()) d = &it->second[0];
            }
            if (le.binding == kPushConstantBinding) {
                // The whole slot, not the layout's push constant range: a
                // shader may declare a larger block than the range covers.
                e.buffer = ring().buffer;
                e.offset = 0;
                e.size = kRingSlot;
            } else if (le.buffer.type != WGPUBufferBindingType_BindingNotUsed &&
                       le.buffer.type != WGPUBufferBindingType_Undefined) {
                if (d && d->buffer) {
                    e.buffer = d->buffer->ensureGpu();
                    e.offset = d->offset;
                    e.size = d->range == VK_WHOLE_SIZE ? d->buffer->size - d->offset : d->range;
                    e.size = (e.size + 3) & ~uint64_t(3);
                } else {
                    e.buffer = dm.buffer;
                    e.size = 65536;
                }
            } else if (le.sampler.type != WGPUSamplerBindingType_BindingNotUsed &&
                       le.sampler.type != WGPUSamplerBindingType_Undefined) {
                WGPUSampler s = nullptr;
                if (d && d->sampler) {
                    s = le.sampler.type == WGPUSamplerBindingType_NonFiltering ? d->sampler->ensureNearest()
                                                                              : d->sampler->ensureGpu();
                }
                if (!s) s = le.sampler.type == WGPUSamplerBindingType_Comparison ? dm.comparison : dm.sampler;
                e.sampler = s;
            } else {
                WGPUTextureView v = (d && d->view) ? d->view->ensureGpu() : nullptr;
                if (!v) v = le.texture.sampleType == WGPUTextureSampleType_Depth ? dm.depth2d : dm.view2d;
                e.textureView = v;
            }
            entries.push_back(e);
        }
        WGPUBindGroupDescriptor desc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        desc.layout = layout;
        desc.entryCount = entries.size();
        desc.entries = entries.data();
        WGPUBindGroup group = wgpuDeviceCreateBindGroup(gpu().device, &desc);
        if (set) {
            // One live group per layout; an older one for the same layout is
            // stale and goes.
            auto& gs = set->groups;
            for (auto it = gs.begin(); it != gs.end(); ) {
                if (it->layout == layout) { transientGroups_.push_back(it->group); it = gs.erase(it); }
                else ++it;
            }
            gs.push_back({layout, set->version, swapFrame, group});
        } else {
            unsetGroups()[layout] = group;
        }
        return group;
    }

    template <typename SetGroup>
    bool bindGroups(VkPipeline_T* p, SetGroup&& setGroup) {
        if (p->pushConstants && pushDirty_) {
            const uint32_t at = ring().push(pushData_.data(), kRingSlot);
            if (at == UINT32_MAX) {
                LOG_ERROR("WebGPU: push constant ring full; draw skipped");
                return false;
            }
            pushOffset_ = at;
            pushDirty_ = false;
        }
        for (uint32_t g = 0; g < p->groupCount; ++g) {
            WGPUBindGroup group = makeGroup(p, g);
            std::vector<uint32_t> offsets;
            for (uint32_t b : p->dynamicBindings[g]) {
                if (b == kPushConstantBinding) offsets.push_back(pushOffset_);
                else {
                    auto it = setDynamic_[g].find(b / 2);
                    offsets.push_back(it == setDynamic_[g].end() ? 0 : it->second);
                }
            }
            if (group != boundGroups_[g] || offsets != boundOffsets_[g]) {
                setGroup(g, group, offsets);
                boundGroups_[g] = group;
                boundOffsets_[g] = std::move(offsets);
            }
        }
        return true;
    }

    /// Before a pipeline's layout exists: which bound textures cannot be
    /// filtered, which WebGPU wants to know in the layout (see VkPipeline_T).
    void noteUnfilterable(VkPipeline_T* p) {
        if (p->gpuLayout) return;
        p->unfilterable.clear();
        for (uint32_t g = 0; g < kMaxBoundSets; ++g) {
            if (!sets_[g]) continue;
            for (const auto& [binding, ds] : sets_[g]->descriptors) {
                if (ds.empty() || !ds[0].view) continue;
                const VkFormat f = ds[0].view->format;
                const FormatInfo fi = formatInfo(f);
                const bool float32 = f == VK_FORMAT_R32_SFLOAT || f == VK_FORMAT_R32G32_SFLOAT ||
                                     f == VK_FORMAT_R32G32B32A32_SFLOAT;
                if (fi.depth || (float32 && !gpu().float32Filterable))
                    p->unfilterable.emplace_back(g, binding);
            }
        }
    }

    bool prepareDraw(bool indexed) {
        VkPipeline_T* p = pipeline_;
        if (!pass_ || !p || p->compute) return false;
        noteUnfilterable(p);
        WGPURenderPipeline rp = p->renderPipeline(biasConstant_, biasSlope_, biasClamp_);
        if (!rp) return false;
        if (rp != boundRender_) {
            wgpuRenderPassEncoderSetPipeline(pass_, rp);
            boundRender_ = rp;
            // Groups set for another pipeline layout may not suit this one.
            boundGroups_ = {};
            for (auto& o : boundOffsets_) o.clear();
        }
        const bool ok = bindGroups(p, [&](uint32_t g, WGPUBindGroup group, const std::vector<uint32_t>& offs) {
            wgpuRenderPassEncoderSetBindGroup(pass_, g, group, offs.size(), offs.data());
        });
        if (!ok) return false;
        for (size_t slot = 0; slot < p->vertexBindings.size(); ++slot) {
            const uint32_t b = p->vertexBindings[slot].binding;
            const VB& vb = b < vertexBuffers_.size() ? vertexBuffers_[b] : VB{};
            if (!vb.buffer) return false;
            if (boundVertex_[slot].buffer != vb.buffer || boundVertex_[slot].offset != vb.offset) {
                wgpuRenderPassEncoderSetVertexBuffer(pass_, static_cast<uint32_t>(slot), vb.buffer->ensureGpu(),
                                                     vb.offset, WGPU_WHOLE_SIZE);
                boundVertex_[slot] = vb;
            }
        }
        if (indexed) {
            if (!indexBuffer_) return false;
            if (indexDirty_) {
                wgpuRenderPassEncoderSetIndexBuffer(pass_, indexBuffer_->ensureGpu(),
                    indexType_ == VK_INDEX_TYPE_UINT32 ? WGPUIndexFormat_Uint32 : WGPUIndexFormat_Uint16,
                    indexOffset_, WGPU_WHOLE_SIZE);
                indexDirty_ = false;
            }
        }
        return true;
    }

    bool prepareDispatch() {
        VkPipeline_T* p = pipeline_;
        if (!p || !p->compute) return false;
        if (pass_) endPasses();
        noteUnfilterable(p);
        if (!cpass_) {
            cpass_ = wgpuCommandEncoderBeginComputePass(encoder(), nullptr);
            encoded_ = true;
        }
        WGPUComputePipeline cp = p->computePipeline();
        if (!cp) return false;
        if (cp != boundCompute_) {
            wgpuComputePassEncoderSetPipeline(cpass_, cp);
            boundCompute_ = cp;
            boundGroups_ = {};
            for (auto& o : boundOffsets_) o.clear();
        }
        return bindGroups(p, [&](uint32_t g, WGPUBindGroup group, const std::vector<uint32_t>& offs) {
            wgpuComputePassEncoderSetBindGroup(cpass_, g, group, offs.size(), offs.data());
        });
    }

    // Transfers ---------------------------------------------------------------------------

    void copyBuffer(const c::CopyBuffer& x) {
        if (const uint8_t* host = x.src->hostData()) {
            beforeQueueWrite();
            WGPUBuffer dst = x.dst->ensureGpu();
            for (const auto& r : x.regions) {
                const uint64_t size = (r.size + 3) & ~uint64_t(3);
                wgpuQueueWriteBuffer(gpu().queue, dst, r.dstOffset, host + r.srcOffset, size);
            }
            return;
        }
        endPasses();
        for (const auto& r : x.regions) {
            wgpuCommandEncoderCopyBufferToBuffer(encoder(), x.src->ensureGpu(), r.srcOffset,
                                                 x.dst->ensureGpu(), r.dstOffset, (r.size + 3) & ~uint64_t(3));
        }
        encoded_ = true;
        if (x.dst->hostData()) warnReadback("vkCmdCopyBuffer");
    }

    void copyBufferToImage(const c::CopyBufferToImage& x) {
        const uint8_t* host = x.src->hostData();
        const FormatInfo fi = formatInfo(x.dst->format);
        if (!host) {
            LOG_ERROR("WebGPU: buffer-to-image copy from a GPU-only buffer is not supported");
            return;
        }
        beforeQueueWrite();
        WGPUTexture tex = x.dst->ensureGpu();
        if (!tex) return;
        for (const auto& r : x.regions) {
            const uint32_t rowTexels = r.bufferRowLength ? r.bufferRowLength : r.imageExtent.width;
            const uint32_t rows = r.bufferImageHeight ? r.bufferImageHeight : r.imageExtent.height;
            const uint32_t bd = fi.blockDim;
            WGPUTexelCopyBufferLayout layout = WGPU_TEXEL_COPY_BUFFER_LAYOUT_INIT;
            layout.offset = 0;
            layout.bytesPerRow = (rowTexels + bd - 1) / bd * fi.blockBytes;
            layout.rowsPerImage = (rows + bd - 1) / bd;
            const uint32_t depth = std::max(r.imageExtent.depth, r.imageSubresource.layerCount);
            const size_t size = size_t(layout.bytesPerRow) * layout.rowsPerImage * depth;
            WGPUTexelCopyTextureInfo dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
            dst.texture = tex;
            dst.mipLevel = r.imageSubresource.mipLevel;
            dst.origin = {uint32_t(r.imageOffset.x), uint32_t(r.imageOffset.y),
                          x.dst->type == VK_IMAGE_TYPE_3D ? uint32_t(r.imageOffset.z)
                                                          : r.imageSubresource.baseArrayLayer};
            dst.aspect = copyAspect(r.imageSubresource.aspectMask, fi);
            // A block-compressed copy covers whole blocks, even where the mip
            // is smaller than one.
            WGPUExtent3D extent = {(r.imageExtent.width + bd - 1) / bd * bd,
                                   (r.imageExtent.height + bd - 1) / bd * bd, depth};
            wgpuQueueWriteTexture(gpu().queue, &dst, host + r.bufferOffset, size, &layout, &extent);
        }
    }

    void copyImageToBuffer(const c::CopyImageToBuffer&) {
        warnReadback("vkCmdCopyImageToBuffer");
    }

    static void warnReadback(const char* what) {
        static bool said = false;
        if (!said) {
            said = true;
            LOG_WARNING("WebGPU: ", what, " into host memory - GPU readback is not implemented yet; "
                        "the CPU keeps reading what it last wrote");
        }
    }

    void copyImage(const c::CopyImage& x) {
        endPasses();
        const FormatInfo sf = formatInfo(x.src->format), df = formatInfo(x.dst->format);
        for (const auto& r : x.regions) {
            WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT, dst = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
            src.texture = x.src->ensureGpu();
            src.mipLevel = r.srcSubresource.mipLevel;
            src.origin = {uint32_t(r.srcOffset.x), uint32_t(r.srcOffset.y),
                          x.src->type == VK_IMAGE_TYPE_3D ? uint32_t(r.srcOffset.z) : r.srcSubresource.baseArrayLayer};
            src.aspect = copyAspect(r.srcSubresource.aspectMask, sf);
            dst.texture = x.dst->ensureGpu();
            dst.mipLevel = r.dstSubresource.mipLevel;
            dst.origin = {uint32_t(r.dstOffset.x), uint32_t(r.dstOffset.y),
                          x.dst->type == VK_IMAGE_TYPE_3D ? uint32_t(r.dstOffset.z) : r.dstSubresource.baseArrayLayer};
            dst.aspect = copyAspect(r.dstSubresource.aspectMask, df);
            if (!src.texture || !dst.texture) continue;
            WGPUExtent3D extent = {r.extent.width, r.extent.height,
                                   std::max(r.extent.depth, r.srcSubresource.layerCount)};
            wgpuCommandEncoderCopyTextureToTexture(encoder(), &src, &dst, &extent);
            encoded_ = true;
        }
    }

    void blitImage(const c::BlitImage& x) {
        endPasses();
        const FormatInfo df = formatInfo(x.dst->format);
        const FormatInfo sf = formatInfo(x.src->format);
        if (df.depth || df.blockDim != 1 || sf.depth) {
            LOG_WARNING("WebGPU: vkCmdBlitImage on a depth or compressed image is not supported");
            return;
        }
        Blitter& bl = blitter();
        WGPURenderPipeline pipe = bl.pipeline(df.wgpu);
        for (const auto& r : x.regions) {
            const uint32_t sm = r.srcSubresource.mipLevel, dm = r.dstSubresource.mipLevel;
            const float sw = float(std::max(1u, x.src->extent.width >> sm));
            const float sh = float(std::max(1u, x.src->extent.height >> sm));
            const int32_t dx0 = std::min(r.dstOffsets[0].x, r.dstOffsets[1].x);
            const int32_t dy0 = std::min(r.dstOffsets[0].y, r.dstOffsets[1].y);
            const int32_t dx1 = std::max(r.dstOffsets[0].x, r.dstOffsets[1].x);
            const int32_t dy1 = std::max(r.dstOffsets[0].y, r.dstOffsets[1].y);
            for (uint32_t l = 0; l < r.srcSubresource.layerCount; ++l) {
                WGPUTextureView sv_ = singleView(x.src, sm, r.srcSubresource.baseArrayLayer + l, sf.wgpu);
                WGPUTextureView dv = singleView(x.dst, dm, r.dstSubresource.baseArrayLayer + l, df.wgpu);
                transientViews_.push_back(sv_);
                transientViews_.push_back(dv);
                const float params[4] = {r.srcOffsets[0].x / sw, r.srcOffsets[0].y / sh,
                                         (r.srcOffsets[1].x - r.srcOffsets[0].x) / sw,
                                         (r.srcOffsets[1].y - r.srcOffsets[0].y) / sh};
                const uint32_t at = ring().push(params, sizeof(params));
                if (at == UINT32_MAX) return;
                WGPUBindGroupEntry e[3] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT,
                                           WGPU_BIND_GROUP_ENTRY_INIT};
                e[0].binding = 0; e[0].textureView = sv_;
                e[1].binding = 1; e[1].sampler = x.filter == VK_FILTER_LINEAR ? bl.linear : bl.nearest;
                e[2].binding = 2; e[2].buffer = ring().buffer; e[2].size = 16;
                WGPUBindGroupDescriptor bd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
                bd.layout = bl.layout;
                bd.entryCount = 3;
                bd.entries = e;
                WGPUBindGroup group = wgpuDeviceCreateBindGroup(gpu().device, &bd);
                transientGroups_.push_back(group);

                WGPURenderPassColorAttachment ca = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
                ca.view = dv;
                ca.loadOp = WGPULoadOp_Load;
                ca.storeOp = WGPUStoreOp_Store;
                WGPURenderPassDescriptor pd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
                pd.colorAttachmentCount = 1;
                pd.colorAttachments = &ca;
                WGPURenderPassEncoder p = wgpuCommandEncoderBeginRenderPass(encoder(), &pd);
                wgpuRenderPassEncoderSetPipeline(p, pipe);
                wgpuRenderPassEncoderSetBindGroup(p, 0, group, 1, &at);
                wgpuRenderPassEncoderSetViewport(p, float(dx0), float(dy0), float(dx1 - dx0),
                                                 float(dy1 - dy0), 0.0f, 1.0f);
                wgpuRenderPassEncoderDraw(p, 3, 1, 0, 0);
                wgpuRenderPassEncoderEnd(p);
                wgpuRenderPassEncoderRelease(p);
                encoded_ = true;
            }
        }
    }

    void clearColor(const c::ClearColorImage& x) {
        endPasses();
        const FormatInfo fi = formatInfo(x.image->format);
        for (const auto& r : x.ranges) {
            const uint32_t levels = r.levelCount == VK_REMAINING_MIP_LEVELS ? x.image->mipLevels - r.baseMipLevel : r.levelCount;
            const uint32_t layers = r.layerCount == VK_REMAINING_ARRAY_LAYERS ? x.image->arrayLayers - r.baseArrayLayer : r.layerCount;
            for (uint32_t m = 0; m < levels; ++m) {
                for (uint32_t l = 0; l < layers; ++l) {
                    WGPUTextureView v = singleView(x.image, r.baseMipLevel + m, r.baseArrayLayer + l, fi.wgpu);
                    transientViews_.push_back(v);
                    WGPURenderPassColorAttachment ca = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
                    ca.view = v;
                    ca.loadOp = WGPULoadOp_Clear;
                    ca.storeOp = WGPUStoreOp_Store;
                    ca.clearValue = {x.color.float32[0], x.color.float32[1], x.color.float32[2], x.color.float32[3]};
                    WGPURenderPassDescriptor pd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
                    pd.colorAttachmentCount = 1;
                    pd.colorAttachments = &ca;
                    WGPURenderPassEncoder p = wgpuCommandEncoderBeginRenderPass(encoder(), &pd);
                    wgpuRenderPassEncoderEnd(p);
                    wgpuRenderPassEncoderRelease(p);
                    encoded_ = true;
                }
            }
        }
    }

    void clearDepth(const c::ClearDepthStencilImage& x) {
        endPasses();
        const FormatInfo fi = formatInfo(x.image->format);
        for (const auto& r : x.ranges) {
            const uint32_t levels = r.levelCount == VK_REMAINING_MIP_LEVELS ? x.image->mipLevels - r.baseMipLevel : r.levelCount;
            for (uint32_t m = 0; m < levels; ++m) {
                WGPUTextureView v = singleView(x.image, r.baseMipLevel + m, r.baseArrayLayer, fi.wgpu);
                transientViews_.push_back(v);
                WGPURenderPassDepthStencilAttachment da = WGPU_RENDER_PASS_DEPTH_STENCIL_ATTACHMENT_INIT;
                da.view = v;
                da.depthLoadOp = WGPULoadOp_Clear;
                da.depthStoreOp = WGPUStoreOp_Store;
                da.depthClearValue = x.value.depth;
                if (fi.stencil) {
                    da.stencilLoadOp = WGPULoadOp_Clear;
                    da.stencilStoreOp = WGPUStoreOp_Store;
                    da.stencilClearValue = x.value.stencil;
                }
                WGPURenderPassDescriptor pd = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
                pd.depthStencilAttachment = &da;
                WGPURenderPassEncoder p = wgpuCommandEncoderBeginRenderPass(encoder(), &pd);
                wgpuRenderPassEncoderEnd(p);
                wgpuRenderPassEncoderRelease(p);
                encoded_ = true;
            }
        }
    }

    void fillBuffer(const c::FillBuffer& x) {
        const uint64_t size = x.size == VK_WHOLE_SIZE ? x.buffer->size - x.offset : x.size;
        if (x.data == 0) {
            endPasses();
            wgpuCommandEncoderClearBuffer(encoder(), x.buffer->ensureGpu(), x.offset, size & ~uint64_t(3));
            encoded_ = true;
            return;
        }
        beforeQueueWrite();
        std::vector<uint32_t> pattern(size / 4, x.data);
        wgpuQueueWriteBuffer(gpu().queue, x.buffer->ensureGpu(), x.offset, pattern.data(), pattern.size() * 4);
    }
};

} // namespace

namespace wgpuvk {

void replay(const std::vector<VkCommandBuffer_T*>& buffers) {
    Replayer r;
    r.run(buffers);
}

} // namespace wgpuvk
