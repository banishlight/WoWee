// Descriptor sets, render passes, framebuffers and pipelines.
//
// A WebGPU pipeline is built the first time it is drawn with, on the main
// thread, from the Vulkan state kept here. Its bind group layouts come from
// the shaders' own reflection (tools/wasm_shaders.py), not from the Vulkan
// descriptor set layouts: the reflection says exactly which bindings a
// shader reads and as what, which WebGPU insists on and Vulkan does not.

#include "wgpu_layer.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>

#include "core/logger.hpp"

using namespace wgpuvk;

namespace {

WGPUStringView sv(const std::string& s) { return {s.c_str(), s.size()}; }

WGPUBlendFactor blendFactor(VkBlendFactor f) {
    switch (f) {
    case VK_BLEND_FACTOR_ZERO:                     return WGPUBlendFactor_Zero;
    case VK_BLEND_FACTOR_ONE:                      return WGPUBlendFactor_One;
    case VK_BLEND_FACTOR_SRC_COLOR:                return WGPUBlendFactor_Src;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR:      return WGPUBlendFactor_OneMinusSrc;
    case VK_BLEND_FACTOR_DST_COLOR:                return WGPUBlendFactor_Dst;
    case VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR:      return WGPUBlendFactor_OneMinusDst;
    case VK_BLEND_FACTOR_SRC_ALPHA:                return WGPUBlendFactor_SrcAlpha;
    case VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA:      return WGPUBlendFactor_OneMinusSrcAlpha;
    case VK_BLEND_FACTOR_DST_ALPHA:                return WGPUBlendFactor_DstAlpha;
    case VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA:      return WGPUBlendFactor_OneMinusDstAlpha;
    case VK_BLEND_FACTOR_CONSTANT_COLOR:
    case VK_BLEND_FACTOR_CONSTANT_ALPHA:           return WGPUBlendFactor_Constant;
    case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR:
    case VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA: return WGPUBlendFactor_OneMinusConstant;
    case VK_BLEND_FACTOR_SRC_ALPHA_SATURATE:       return WGPUBlendFactor_SrcAlphaSaturated;
    default:                                       return WGPUBlendFactor_One;
    }
}

WGPUBlendOperation blendOp(VkBlendOp op) {
    switch (op) {
    case VK_BLEND_OP_SUBTRACT:         return WGPUBlendOperation_Subtract;
    case VK_BLEND_OP_REVERSE_SUBTRACT: return WGPUBlendOperation_ReverseSubtract;
    case VK_BLEND_OP_MIN:              return WGPUBlendOperation_Min;
    case VK_BLEND_OP_MAX:              return WGPUBlendOperation_Max;
    default:                           return WGPUBlendOperation_Add;
    }
}

WGPUCompareFunction compareFn(VkCompareOp op) {
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

WGPUStencilOperation stencilOp(VkStencilOp op) {
    switch (op) {
    case VK_STENCIL_OP_ZERO:                return WGPUStencilOperation_Zero;
    case VK_STENCIL_OP_REPLACE:             return WGPUStencilOperation_Replace;
    case VK_STENCIL_OP_INCREMENT_AND_CLAMP: return WGPUStencilOperation_IncrementClamp;
    case VK_STENCIL_OP_DECREMENT_AND_CLAMP: return WGPUStencilOperation_DecrementClamp;
    case VK_STENCIL_OP_INVERT:              return WGPUStencilOperation_Invert;
    case VK_STENCIL_OP_INCREMENT_AND_WRAP:  return WGPUStencilOperation_IncrementWrap;
    case VK_STENCIL_OP_DECREMENT_AND_WRAP:  return WGPUStencilOperation_DecrementWrap;
    default:                                return WGPUStencilOperation_Keep;
    }
}

WGPUStencilFaceState stencilFace(const VkStencilOpState& s) {
    WGPUStencilFaceState f = WGPU_STENCIL_FACE_STATE_INIT;
    f.compare = compareFn(s.compareOp);
    f.failOp = stencilOp(s.failOp);
    f.depthFailOp = stencilOp(s.depthFailOp);
    f.passOp = stencilOp(s.passOp);
    return f;
}

/// A bind group layout entry, built from what the shaders say they read.
WGPUBindGroupLayoutEntry layoutEntry(const ShaderResource& r, WGPUShaderStage stage, bool dynamic) {
    WGPUBindGroupLayoutEntry e = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    e.binding = r.binding;
    e.visibility = stage;
    using K = ShaderResource::Kind;
    switch (r.kind) {
    case K::Uniform:
        e.buffer.type = WGPUBufferBindingType_Uniform;
        e.buffer.hasDynamicOffset = dynamic;
        break;
    case K::Storage:
        e.buffer.type = WGPUBufferBindingType_Storage;
        e.buffer.hasDynamicOffset = dynamic;
        break;
    case K::ReadOnlyStorage:
        e.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
        e.buffer.hasDynamicOffset = dynamic;
        break;
    case K::Sampler:
        e.sampler.type = WGPUSamplerBindingType_Filtering;
        break;
    case K::ComparisonSampler:
        e.sampler.type = WGPUSamplerBindingType_Comparison;
        break;
    case K::Texture: {
        const std::string& t = r.type;
        e.texture.multisampled = t.find("multisampled") != std::string::npos;
        if (t.find("depth") != std::string::npos)      e.texture.sampleType = WGPUTextureSampleType_Depth;
        else if (t.find("<u32") != std::string::npos)  e.texture.sampleType = WGPUTextureSampleType_Uint;
        else if (t.find("<i32") != std::string::npos)  e.texture.sampleType = WGPUTextureSampleType_Sint;
        else if (e.texture.multisampled)               e.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
        else                                           e.texture.sampleType = WGPUTextureSampleType_Float;
        if (t.find("cube_array") != std::string::npos)      e.texture.viewDimension = WGPUTextureViewDimension_CubeArray;
        else if (t.find("cube") != std::string::npos)       e.texture.viewDimension = WGPUTextureViewDimension_Cube;
        else if (t.find("2d_array") != std::string::npos)   e.texture.viewDimension = WGPUTextureViewDimension_2DArray;
        else if (t.find("3d") != std::string::npos)         e.texture.viewDimension = WGPUTextureViewDimension_3D;
        else if (t.find("1d") != std::string::npos)         e.texture.viewDimension = WGPUTextureViewDimension_1D;
        else                                                e.texture.viewDimension = WGPUTextureViewDimension_2D;
        break;
    }
    case K::StorageTexture: {
        const std::string& t = r.type;   // texture_storage_2d<r32float,read_write>
        const auto lt = t.find('<'), comma = t.find(',');
        const std::string fmt = t.substr(lt + 1, comma - lt - 1);
        e.storageTexture.access = t.find("read_write") != std::string::npos
            ? WGPUStorageTextureAccess_ReadWrite
            : t.find(",read>") != std::string::npos ? WGPUStorageTextureAccess_ReadOnly
                                                    : WGPUStorageTextureAccess_WriteOnly;
        e.storageTexture.format = fmt == "r32float"    ? WGPUTextureFormat_R32Float
                                : fmt == "rg16float"   ? WGPUTextureFormat_RG16Float
                                : fmt == "rgba16float" ? WGPUTextureFormat_RGBA16Float
                                : fmt == "rgba8unorm"  ? WGPUTextureFormat_RGBA8Unorm
                                : fmt == "r32uint"     ? WGPUTextureFormat_R32Uint
                                : fmt == "rg32float"   ? WGPUTextureFormat_RG32Float
                                                       : WGPUTextureFormat_RGBA32Float;
        e.storageTexture.viewDimension = t.find("2d_array") != std::string::npos
            ? WGPUTextureViewDimension_2DArray : WGPUTextureViewDimension_2D;
        break;
    }
    }
    return e;
}

/// Reports a pipeline WebGPU refused, once, with the shaders it was built
/// from. WebGPU only says why at creation - every later use of the pipeline
/// just says it is invalid - so without this the reason is one line lost
/// among thousands.
void pushPipelineScope() {
    wgpuDevicePushErrorScope(gpu().device, WGPUErrorFilter_Validation);
}

void popPipelineScope(const std::string& name, std::shared_ptr<std::atomic<bool>> refused) {
    struct Scope {
        std::string name;
        std::shared_ptr<std::atomic<bool>> refused;
    };
    WGPUPopErrorScopeCallbackInfo cb = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = [](WGPUPopErrorScopeStatus, WGPUErrorType type, WGPUStringView message,
                     void* userdata, void*) {
        auto* scope = static_cast<Scope*>(userdata);
        if (type != WGPUErrorType_NoError) {
            scope->refused->store(true);
            LOG_ERROR("WebGPU pipeline ", scope->name, " refused (its draws are skipped): ",
                      std::string(message.data, message.length));
        }
        delete scope;
    };
    cb.userdata1 = new Scope{name, std::move(refused)};
    wgpuDevicePopErrorScope(gpu().device, cb);
}

/// Bind group layouts, shared between pipelines whose groups read the same.
WGPUBindGroupLayout cachedGroupLayout(const std::vector<WGPUBindGroupLayoutEntry>& entries) {
    static std::map<std::string, WGPUBindGroupLayout> cache;
    // Field by field: the structs themselves carry padding and pointers.
    std::string key;
    for (const auto& e : entries) {
        const uint64_t fields[] = {
            e.binding, static_cast<uint64_t>(e.visibility),
            static_cast<uint64_t>(e.buffer.type), e.buffer.hasDynamicOffset,
            static_cast<uint64_t>(e.sampler.type),
            static_cast<uint64_t>(e.texture.sampleType),
            static_cast<uint64_t>(e.texture.viewDimension), e.texture.multisampled,
            static_cast<uint64_t>(e.storageTexture.access),
            static_cast<uint64_t>(e.storageTexture.format),
            static_cast<uint64_t>(e.storageTexture.viewDimension)};
        key.append(reinterpret_cast<const char*>(fields), sizeof(fields));
    }
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    WGPUBindGroupLayoutDescriptor desc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    desc.entryCount = entries.size();
    desc.entries = entries.data();
    WGPUBindGroupLayout layout = wgpuDeviceCreateBindGroupLayout(gpu().device, &desc);
    cache.emplace(std::move(key), layout);
    return layout;
}

} // namespace

// Pipeline methods ----------------------------------------------------------------------

const VkDescriptorSetLayout_T::Binding* VkDescriptorSetLayout_T::find(uint32_t b) const {
    for (const auto& x : bindings) if (x.binding == b) return &x;
    return nullptr;
}

VkDescriptorSet_T::~VkDescriptorSet_T() {
    for (auto& g : groups) wgpuBindGroupRelease(g.group);
}

bool VkPipeline_T::ensureLayout() {
    if (gpuLayout || failed) return !failed;

    std::vector<std::pair<const ShaderInfo*, WGPUShaderStage>> stages;
    if (compute) {
        if (compute_ && compute_->shader) stages.push_back({compute_->shader, WGPUShaderStage_Compute});
    } else {
        if (vertex && vertex->shader) stages.push_back({vertex->shader, WGPUShaderStage_Vertex});
        if (fragment && fragment->shader) stages.push_back({fragment->shader, WGPUShaderStage_Fragment});
    }
    if (stages.empty() || (!compute && !(vertex && vertex->shader))) {
        failed = true;
        return false;
    }

    // Per group, binding -> entry, merged across stages.
    std::array<std::map<uint32_t, WGPUBindGroupLayoutEntry>, kMaxBoundSets> groups;
    uint32_t count = static_cast<uint32_t>(layout ? layout->setLayouts.size() : 0);
    for (const auto& [shader, stage] : stages) {
        for (const ShaderResource& r : shader->resources) {
            if (r.group >= kMaxBoundSets) { failed = true; return false; }
            bool dynamic = false;
            if (r.binding == kPushConstantBinding) {
                dynamic = true;
                pushConstants = true;
            } else if (layout && r.group < layout->setLayouts.size()) {
                const auto* b = layout->setLayouts[r.group]->find(r.binding / 2);
                dynamic = b && (b->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC ||
                                b->type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC);
            }
            auto [it, inserted] = groups[r.group].try_emplace(r.binding, layoutEntry(r, stage, dynamic));
            if (!inserted) it->second.visibility |= stage;
            count = std::max(count, r.group + 1);
        }
    }
    groupCount = count;

    // Textures bound unfilterable, and the samplers split off beside them.
    for (const auto& [g, vkBinding] : unfilterable) {
        if (g >= kMaxBoundSets) continue;
        auto tex = groups[g].find(textureBinding(vkBinding));
        if (tex == groups[g].end() || tex->second.texture.sampleType != WGPUTextureSampleType_Float)
            continue;
        tex->second.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
        auto smp = groups[g].find(samplerBinding(vkBinding));
        if (smp != groups[g].end() && smp->second.sampler.type == WGPUSamplerBindingType_Filtering)
            smp->second.sampler.type = WGPUSamplerBindingType_NonFiltering;
    }

    std::vector<WGPUBindGroupLayout> layouts;
    for (uint32_t g = 0; g < groupCount; ++g) {
        std::vector<WGPUBindGroupLayoutEntry> entries;
        for (auto& [binding, e] : groups[g]) {
            entries.push_back(e);
            if (e.buffer.hasDynamicOffset) dynamicBindings[g].push_back(binding);
        }
        groupLayouts[g] = cachedGroupLayout(entries);
        groupEntries[g] = entries;
        layouts.push_back(groupLayouts[g]);
    }
    WGPUPipelineLayoutDescriptor desc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    desc.bindGroupLayoutCount = layouts.size();
    desc.bindGroupLayouts = layouts.data();
    gpuLayout = wgpuDeviceCreatePipelineLayout(gpu().device, &desc);
    return true;
}

WGPURenderPipeline VkPipeline_T::renderPipeline(float biasConstant, float biasSlope, float biasClamp) {
    if (!depthBiasEnable) biasConstant = biasSlope = biasClamp = 0.0f;
    else if (!dynamicDepthBias) {
        biasConstant = depthBiasConstant;
        biasSlope = depthBiasSlope;
        biasClamp = depthBiasClamp;
    }
    for (const auto& v : variants) {
        if (v.constant == biasConstant && v.slope == biasSlope && v.clamp == biasClamp)
            return v.refused->load() ? nullptr : v.pipeline;
    }
    if (!ensureLayout()) return nullptr;
    WGPUShaderModule vs = vertex->ensureGpu();
    WGPUShaderModule fs = (fragment && fragment->shader) ? fragment->ensureGpu() : nullptr;
    if (!vs) { failed = true; return nullptr; }

    WGPURenderPipelineDescriptor desc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    desc.label = sv(vertex->shader->name);
    desc.layout = gpuLayout;

    // Vertex buffers: one WebGPU slot per Vulkan binding, in binding order.
    std::vector<WGPUVertexBufferLayout> buffers;
    std::vector<std::vector<WGPUVertexAttribute>> attributes(vertexBindings.size());
    for (size_t i = 0; i < vertexBindings.size(); ++i) {
        const auto& b = vertexBindings[i];
        for (const auto& a : vertexAttributes) {
            if (a.binding != b.binding) continue;
            WGPUVertexAttribute attr = WGPU_VERTEX_ATTRIBUTE_INIT;
            attr.format = vertexFormat(a.format);
            attr.offset = a.offset;
            attr.shaderLocation = a.location;
            attributes[i].push_back(attr);
        }
        WGPUVertexBufferLayout l = WGPU_VERTEX_BUFFER_LAYOUT_INIT;
        l.arrayStride = b.stride;
        l.stepMode = b.inputRate == VK_VERTEX_INPUT_RATE_INSTANCE
            ? WGPUVertexStepMode_Instance : WGPUVertexStepMode_Vertex;
        l.attributeCount = attributes[i].size();
        l.attributes = attributes[i].data();
        buffers.push_back(l);
    }
    desc.vertex.module = vs;
    desc.vertex.entryPoint = sv(vertexEntry);
    desc.vertex.bufferCount = buffers.size();
    desc.vertex.buffers = buffers.data();

    bool pointOrLine = false;
    switch (topology) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST:
        desc.primitive.topology = WGPUPrimitiveTopology_PointList; pointOrLine = true; break;
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
        desc.primitive.topology = WGPUPrimitiveTopology_LineList; pointOrLine = true; break;
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
        desc.primitive.topology = WGPUPrimitiveTopology_LineStrip; pointOrLine = true; break;
    case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP:
        desc.primitive.topology = WGPUPrimitiveTopology_TriangleStrip; break;
    default:
        desc.primitive.topology = WGPUPrimitiveTopology_TriangleList; break;
    }
    desc.primitive.frontFace = frontFace == VK_FRONT_FACE_CLOCKWISE ? WGPUFrontFace_CW : WGPUFrontFace_CCW;
    desc.primitive.cullMode = cullMode == VK_CULL_MODE_FRONT_BIT ? WGPUCullMode_Front
                            : cullMode == VK_CULL_MODE_BACK_BIT  ? WGPUCullMode_Back
                                                                 : WGPUCullMode_None;

    const VkRenderPass_T* rp = renderPass;
    WGPUDepthStencilState ds = WGPU_DEPTH_STENCIL_STATE_INIT;
    if (rp && rp->depth != VK_ATTACHMENT_UNUSED) {
        const FormatInfo fi = formatInfo(rp->attachments[rp->depth].format);
        ds.format = fi.wgpu;
        if (hasDepthStencil && depthStencil.depthTestEnable) {
            ds.depthWriteEnabled = depthStencil.depthWriteEnable ? WGPUOptionalBool_True : WGPUOptionalBool_False;
            ds.depthCompare = compareFn(depthStencil.depthCompareOp);
        } else {
            ds.depthWriteEnabled = WGPUOptionalBool_False;
            ds.depthCompare = WGPUCompareFunction_Always;
        }
        if (fi.stencil && hasDepthStencil && depthStencil.stencilTestEnable) {
            ds.stencilFront = stencilFace(depthStencil.front);
            ds.stencilBack = stencilFace(depthStencil.back);
            ds.stencilReadMask = depthStencil.front.compareMask;
            ds.stencilWriteMask = depthStencil.front.writeMask;
        }
        if (!pointOrLine) {
            ds.depthBias = static_cast<int32_t>(biasConstant);
            ds.depthBiasSlopeScale = biasSlope;
            ds.depthBiasClamp = biasClamp;
        }
        desc.depthStencil = &ds;
    }

    desc.multisample.count = rp ? sampleCount(rp->samples) : 1;
    // WebGPU allows alpha-to-coverage only with more than one sample. Vulkan
    // allows it with one, where it amounts to a cut at half alpha - close
    // enough to drawing without it that the pipeline is better built than lost.
    desc.multisample.alphaToCoverageEnabled = alphaToCoverage && desc.multisample.count > 1;

    std::vector<WGPUColorTargetState> targets;
    std::vector<WGPUBlendState> blendStates;
    if (rp) {
        targets.reserve(rp->colors.size());
        blendStates.reserve(rp->colors.size());
        for (size_t i = 0; i < rp->colors.size(); ++i) {
            WGPUColorTargetState t = WGPU_COLOR_TARGET_STATE_INIT;
            t.format = formatInfo(rp->attachments[rp->colors[i]].format).wgpu;
            const VkPipelineColorBlendAttachmentState* b =
                i < blends.size() ? &blends[i] : (blends.empty() ? nullptr : &blends[0]);
            if (b) {
                t.writeMask = static_cast<WGPUColorWriteMask>(b->colorWriteMask & 0xF);
                if (b->blendEnable) {
                    WGPUBlendState bs = WGPU_BLEND_STATE_INIT;
                    bs.color = {blendOp(b->colorBlendOp), blendFactor(b->srcColorBlendFactor),
                                blendFactor(b->dstColorBlendFactor)};
                    bs.alpha = {blendOp(b->alphaBlendOp), blendFactor(b->srcAlphaBlendFactor),
                                blendFactor(b->dstAlphaBlendFactor)};
                    // Min and max take no factors in WebGPU.
                    if (bs.color.operation == WGPUBlendOperation_Min || bs.color.operation == WGPUBlendOperation_Max)
                        bs.color.srcFactor = bs.color.dstFactor = WGPUBlendFactor_One;
                    if (bs.alpha.operation == WGPUBlendOperation_Min || bs.alpha.operation == WGPUBlendOperation_Max)
                        bs.alpha.srcFactor = bs.alpha.dstFactor = WGPUBlendFactor_One;
                    blendStates.push_back(bs);
                    t.blend = &blendStates.back();
                }
            }
            targets.push_back(t);
        }
    }
    WGPUFragmentState fsState = WGPU_FRAGMENT_STATE_INIT;
    if (fs) {
        fsState.module = fs;
        fsState.entryPoint = sv(fragmentEntry);
        fsState.targetCount = targets.size();
        fsState.targets = targets.data();
        desc.fragment = &fsState;
    }

    const std::string label = vertex->shader->name +
        (fragment && fragment->shader ? " + " + fragment->shader->name : std::string());
    auto refused = std::make_shared<std::atomic<bool>>(false);
    pushPipelineScope();
    WGPURenderPipeline p = wgpuDeviceCreateRenderPipeline(gpu().device, &desc);
    popPipelineScope(label, refused);
    if (!p) failed = true;
    variants.push_back({biasConstant, biasSlope, biasClamp, p, refused});
    return p;
}

WGPUComputePipeline VkPipeline_T::computePipeline() {
    if (computeRefused->load()) return nullptr;
    if (gpuCompute || failed) return gpuCompute;
    if (!ensureLayout()) return nullptr;
    WGPUShaderModule cs = compute_->ensureGpu();
    if (!cs) { failed = true; return nullptr; }
    WGPUComputePipelineDescriptor desc = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    desc.label = sv(compute_->shader->name);
    desc.layout = gpuLayout;
    desc.compute.module = cs;
    desc.compute.entryPoint = sv(computeEntry);
    pushPipelineScope();
    gpuCompute = wgpuDeviceCreateComputePipeline(gpu().device, &desc);
    popPipelineScope(compute_->shader->name, computeRefused);
    if (!gpuCompute) failed = true;
    return gpuCompute;
}

extern "C" {

// Descriptor sets ---------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorSetLayout(
        VkDevice, const VkDescriptorSetLayoutCreateInfo* ci, const VkAllocationCallbacks*,
        VkDescriptorSetLayout* pLayout) {
    auto* l = new VkDescriptorSetLayout_T();
    for (uint32_t i = 0; i < ci->bindingCount; ++i) {
        const auto& src = ci->pBindings[i];
        VkDescriptorSetLayout_T::Binding b;
        b.binding = src.binding;
        b.type = src.descriptorType;
        b.count = src.descriptorCount;
        if (src.pImmutableSamplers) {
            for (uint32_t s = 0; s < src.descriptorCount; ++s)
                b.immutableSamplers.push_back(src.pImmutableSamplers[s]);
        }
        l->bindings.push_back(std::move(b));
    }
    *pLayout = l;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorSetLayout(VkDevice, VkDescriptorSetLayout, const VkAllocationCallbacks*) {
    // Kept: descriptor sets and pipeline layouts refer to their layouts after
    // the app has let go of them, which Vulkan allows.
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateDescriptorPool(
        VkDevice, const VkDescriptorPoolCreateInfo*, const VkAllocationCallbacks*, VkDescriptorPool* pPool) {
    *pPool = new VkDescriptorPool_T();
    return VK_SUCCESS;
}

static void freeSet(VkDescriptorSet_T* set) {
    releaseLater([set] { delete set; });
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDescriptorPool(VkDevice, VkDescriptorPool pool, const VkAllocationCallbacks*) {
    if (!pool) return;
    for (auto* s : pool->sets) freeSet(s);
    delete pool;
}

VKAPI_ATTR VkResult VKAPI_CALL vkResetDescriptorPool(VkDevice, VkDescriptorPool pool, VkDescriptorPoolResetFlags) {
    for (auto* s : pool->sets) freeSet(s);
    pool->sets.clear();
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkAllocateDescriptorSets(
        VkDevice, const VkDescriptorSetAllocateInfo* ai, VkDescriptorSet* pSets) {
    for (uint32_t i = 0; i < ai->descriptorSetCount; ++i) {
        auto* s = new VkDescriptorSet_T();
        s->layout = ai->pSetLayouts[i];
        s->pool = ai->descriptorPool;
        // Immutable samplers are part of the set from the start.
        for (const auto& b : s->layout->bindings) {
            if (b.immutableSamplers.empty()) continue;
            auto& ds = s->descriptors[b.binding];
            ds.resize(b.immutableSamplers.size());
            for (size_t k = 0; k < ds.size(); ++k) ds[k].sampler = b.immutableSamplers[k];
        }
        ai->descriptorPool->sets.push_back(s);
        pSets[i] = s;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkFreeDescriptorSets(
        VkDevice, VkDescriptorPool pool, uint32_t count, const VkDescriptorSet* sets) {
    for (uint32_t i = 0; i < count; ++i) {
        if (!sets[i]) continue;
        auto& v = pool->sets;
        v.erase(std::remove(v.begin(), v.end(), sets[i]), v.end());
        freeSet(sets[i]);
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkUpdateDescriptorSets(
        VkDevice, uint32_t writeCount, const VkWriteDescriptorSet* writes,
        uint32_t copyCount, const VkCopyDescriptorSet* copies) {
    for (uint32_t i = 0; i < writeCount; ++i) {
        const auto& w = writes[i];
        VkDescriptorSet_T* set = w.dstSet;
        auto& ds = set->descriptors[w.dstBinding];
        const auto* layoutBinding = set->layout->find(w.dstBinding);
        if (ds.size() < w.dstArrayElement + w.descriptorCount)
            ds.resize(w.dstArrayElement + w.descriptorCount);
        for (uint32_t k = 0; k < w.descriptorCount; ++k) {
            auto& d = ds[w.dstArrayElement + k];
            switch (w.descriptorType) {
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
                d.buffer = w.pBufferInfo[k].buffer;
                d.offset = w.pBufferInfo[k].offset;
                d.range = w.pBufferInfo[k].range;
                break;
            case VK_DESCRIPTOR_TYPE_SAMPLER:
                d.sampler = w.pImageInfo[k].sampler;
                break;
            default:
                d.view = w.pImageInfo[k].imageView;
                if (w.pImageInfo[k].sampler &&
                    (!layoutBinding || layoutBinding->immutableSamplers.empty()))
                    d.sampler = w.pImageInfo[k].sampler;
                break;
            }
        }
        ++set->version;
    }
    for (uint32_t i = 0; i < copyCount; ++i) {
        const auto& c = copies[i];
        const auto& src = c.srcSet->descriptors[c.srcBinding];
        auto& dst = c.dstSet->descriptors[c.dstBinding];
        if (dst.size() < c.dstArrayElement + c.descriptorCount)
            dst.resize(c.dstArrayElement + c.descriptorCount);
        for (uint32_t k = 0; k < c.descriptorCount && c.srcArrayElement + k < src.size(); ++k)
            dst[c.dstArrayElement + k] = src[c.srcArrayElement + k];
        ++c.dstSet->version;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreatePipelineLayout(
        VkDevice, const VkPipelineLayoutCreateInfo* ci, const VkAllocationCallbacks*, VkPipelineLayout* pLayout) {
    auto* l = new VkPipelineLayout_T();
    for (uint32_t i = 0; i < ci->setLayoutCount; ++i) l->setLayouts.push_back(ci->pSetLayouts[i]);
    for (uint32_t i = 0; i < ci->pushConstantRangeCount; ++i) {
        const auto& r = ci->pPushConstantRanges[i];
        l->pushConstantBytes = std::max(l->pushConstantBytes, r.offset + r.size);
    }
    *pLayout = l;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyPipelineLayout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*) {
    // Kept, as descriptor set layouts are: pipelines refer to them.
}

// Render passes and framebuffers ----------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL vkCreateRenderPass(
        VkDevice, const VkRenderPassCreateInfo* ci, const VkAllocationCallbacks*, VkRenderPass* pPass) {
    auto* rp = new VkRenderPass_T();
    rp->attachments.assign(ci->pAttachments, ci->pAttachments + ci->attachmentCount);
    if (ci->subpassCount > 1)
        LOG_WARNING("WebGPU: render pass with ", ci->subpassCount, " subpasses; only the first is used");
    const VkSubpassDescription& sp = ci->pSubpasses[0];
    for (uint32_t i = 0; i < sp.colorAttachmentCount; ++i) {
        rp->colors.push_back(sp.pColorAttachments[i].attachment);
        rp->resolves.push_back(sp.pResolveAttachments ? sp.pResolveAttachments[i].attachment
                                                      : VK_ATTACHMENT_UNUSED);
    }
    if (sp.pDepthStencilAttachment) rp->depth = sp.pDepthStencilAttachment->attachment;
    for (uint32_t c : rp->colors)
        if (c != VK_ATTACHMENT_UNUSED) rp->samples = rp->attachments[c].samples;
    if (rp->depth != VK_ATTACHMENT_UNUSED) rp->samples = rp->attachments[rp->depth].samples;
    *pPass = rp;
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateRenderPass2(
        VkDevice, const VkRenderPassCreateInfo2* ci, const VkAllocationCallbacks*, VkRenderPass* pPass) {
    auto* rp = new VkRenderPass_T();
    for (uint32_t i = 0; i < ci->attachmentCount; ++i) {
        const auto& a = ci->pAttachments[i];
        rp->attachments.push_back({a.flags, a.format, a.samples, a.loadOp, a.storeOp,
                                   a.stencilLoadOp, a.stencilStoreOp, a.initialLayout, a.finalLayout});
    }
    const VkSubpassDescription2& sp = ci->pSubpasses[0];
    for (uint32_t i = 0; i < sp.colorAttachmentCount; ++i) {
        rp->colors.push_back(sp.pColorAttachments[i].attachment);
        rp->resolves.push_back(sp.pResolveAttachments ? sp.pResolveAttachments[i].attachment
                                                      : VK_ATTACHMENT_UNUSED);
    }
    if (sp.pDepthStencilAttachment) rp->depth = sp.pDepthStencilAttachment->attachment;
    // A depth resolve (VkSubpassDescriptionDepthStencilResolve) has no WebGPU
    // equivalent. The renderer only asks for one when the device reports
    // support, and this device reports none.
    for (uint32_t c : rp->colors)
        if (c != VK_ATTACHMENT_UNUSED) rp->samples = rp->attachments[c].samples;
    if (rp->depth != VK_ATTACHMENT_UNUSED) rp->samples = rp->attachments[rp->depth].samples;
    *pPass = rp;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyRenderPass(VkDevice, VkRenderPass, const VkAllocationCallbacks*) {
    // Kept: pipelines built lazily read their render pass's formats.
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateFramebuffer(
        VkDevice, const VkFramebufferCreateInfo* ci, const VkAllocationCallbacks*, VkFramebuffer* pFb) {
    auto* fb = new VkFramebuffer_T();
    fb->attachments.assign(ci->pAttachments, ci->pAttachments + ci->attachmentCount);
    fb->width = ci->width;
    fb->height = ci->height;
    *pFb = fb;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyFramebuffer(VkDevice, VkFramebuffer fb, const VkAllocationCallbacks*) {
    if (fb) releaseLater([fb] { delete fb; });
}

// Pipelines -------------------------------------------------------------------------------

VKAPI_ATTR VkResult VKAPI_CALL vkCreateGraphicsPipelines(
        VkDevice, VkPipelineCache, uint32_t count, const VkGraphicsPipelineCreateInfo* infos,
        const VkAllocationCallbacks*, VkPipeline* pPipelines) {
    for (uint32_t i = 0; i < count; ++i) {
        const auto& ci = infos[i];
        auto* p = new VkPipeline_T();
        p->layout = ci.layout;
        p->renderPass = ci.renderPass;
        for (uint32_t s = 0; s < ci.stageCount; ++s) {
            const auto& st = ci.pStages[s];
            if (st.stage == VK_SHADER_STAGE_VERTEX_BIT) {
                p->vertex = st.module;
                p->vertexEntry = st.pName;
            } else if (st.stage == VK_SHADER_STAGE_FRAGMENT_BIT) {
                p->fragment = st.module;
                p->fragmentEntry = st.pName;
            }
        }
        if (const auto* vi = ci.pVertexInputState) {
            p->vertexBindings.assign(vi->pVertexBindingDescriptions,
                                     vi->pVertexBindingDescriptions + vi->vertexBindingDescriptionCount);
            std::sort(p->vertexBindings.begin(), p->vertexBindings.end(),
                      [](const auto& a, const auto& b) { return a.binding < b.binding; });
            p->vertexAttributes.assign(vi->pVertexAttributeDescriptions,
                                       vi->pVertexAttributeDescriptions + vi->vertexAttributeDescriptionCount);
        }
        if (ci.pInputAssemblyState) p->topology = ci.pInputAssemblyState->topology;
        if (const auto* rs = ci.pRasterizationState) {
            p->cullMode = rs->cullMode;
            p->frontFace = rs->frontFace;
            p->depthBiasEnable = rs->depthBiasEnable;
            p->depthBiasConstant = rs->depthBiasConstantFactor;
            p->depthBiasSlope = rs->depthBiasSlopeFactor;
            p->depthBiasClamp = rs->depthBiasClamp;
            p->depthClamp = rs->depthClampEnable;
        }
        if (ci.pMultisampleState) p->alphaToCoverage = ci.pMultisampleState->alphaToCoverageEnable;
        if (ci.pDepthStencilState) {
            p->depthStencil = *ci.pDepthStencilState;
            p->depthStencil.pNext = nullptr;
            p->hasDepthStencil = true;
        }
        if (const auto* cb = ci.pColorBlendState) {
            p->blends.assign(cb->pAttachments, cb->pAttachments + cb->attachmentCount);
        }
        bool dynViewport = false, dynScissor = false;
        if (const auto* dy = ci.pDynamicState) {
            for (uint32_t d = 0; d < dy->dynamicStateCount; ++d) {
                const VkDynamicState st = dy->pDynamicStates[d];
                if (st == VK_DYNAMIC_STATE_DEPTH_BIAS) p->dynamicDepthBias = true;
                if (st == VK_DYNAMIC_STATE_VIEWPORT) dynViewport = true;
                if (st == VK_DYNAMIC_STATE_SCISSOR) dynScissor = true;
            }
        }
        if (const auto* vp = ci.pViewportState) {
            if (!dynViewport && vp->pViewports && vp->viewportCount > 0) {
                p->staticViewport = true;
                p->viewport = vp->pViewports[0];
            }
            if (!dynScissor && vp->pScissors && vp->scissorCount > 0) {
                p->staticScissor = true;
                p->scissor = vp->pScissors[0];
            }
        }
        pPipelines[i] = p;
        // ?eager (WOWEE_EAGER_PIPELINES): build it now rather than at its
        // first draw, so a pipeline WebGPU refuses shows up at startup - in
        // a test run that never gets as far as drawing with it.
        static const bool eager = std::getenv("WOWEE_EAGER_PIPELINES") != nullptr;
        if (eager && onMainThread() && p->vertex && p->vertex->shader)
            p->renderPipeline(p->depthBiasConstant, p->depthBiasSlope, p->depthBiasClamp);
    }
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL vkCreateComputePipelines(
        VkDevice, VkPipelineCache, uint32_t count, const VkComputePipelineCreateInfo* infos,
        const VkAllocationCallbacks*, VkPipeline* pPipelines) {
    for (uint32_t i = 0; i < count; ++i) {
        auto* p = new VkPipeline_T();
        p->compute = true;
        p->layout = infos[i].layout;
        p->compute_ = infos[i].stage.module;
        p->computeEntry = infos[i].stage.pName;
        pPipelines[i] = p;
    }
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL vkDestroyPipeline(VkDevice, VkPipeline pipeline, const VkAllocationCallbacks*) {
    if (!pipeline) return;
    releaseLater([pipeline] {
        for (auto& v : pipeline->variants) if (v.pipeline) wgpuRenderPipelineRelease(v.pipeline);
        if (pipeline->gpuCompute) wgpuComputePipelineRelease(pipeline->gpuCompute);
        if (pipeline->gpuLayout) wgpuPipelineLayoutRelease(pipeline->gpuLayout);
        delete pipeline;
    });
}

} // extern "C"
