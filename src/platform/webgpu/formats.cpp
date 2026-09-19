// Vulkan formats as WebGPU names them.
//
// Only what a WebGPU texture can be. Formats with no WebGPU equivalent - the
// three-channel ones, mostly - answer Undefined, and
// vkGetPhysicalDeviceFormatProperties reports them as unsupported so the
// renderer's own fallbacks choose something else.

#include "wgpu_layer.hpp"

namespace wgpuvk {

FormatInfo formatInfo(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R8_UNORM:               return {WGPUTextureFormat_R8Unorm, 1};
    case VK_FORMAT_R8_SNORM:               return {WGPUTextureFormat_R8Snorm, 1};
    case VK_FORMAT_R8_UINT:                return {WGPUTextureFormat_R8Uint, 1};
    case VK_FORMAT_R8_SINT:                return {WGPUTextureFormat_R8Sint, 1};
    case VK_FORMAT_R8G8_UNORM:             return {WGPUTextureFormat_RG8Unorm, 2};
    case VK_FORMAT_R8G8_SNORM:             return {WGPUTextureFormat_RG8Snorm, 2};
    case VK_FORMAT_R8G8_UINT:              return {WGPUTextureFormat_RG8Uint, 2};
    case VK_FORMAT_R8G8B8A8_UNORM:         return {WGPUTextureFormat_RGBA8Unorm, 4};
    case VK_FORMAT_R8G8B8A8_SRGB:          return {WGPUTextureFormat_RGBA8UnormSrgb, 4};
    case VK_FORMAT_R8G8B8A8_SNORM:         return {WGPUTextureFormat_RGBA8Snorm, 4};
    case VK_FORMAT_R8G8B8A8_UINT:          return {WGPUTextureFormat_RGBA8Uint, 4};
    case VK_FORMAT_R8G8B8A8_SINT:          return {WGPUTextureFormat_RGBA8Sint, 4};
    case VK_FORMAT_B8G8R8A8_UNORM:         return {WGPUTextureFormat_BGRA8Unorm, 4};
    case VK_FORMAT_B8G8R8A8_SRGB:          return {WGPUTextureFormat_BGRA8UnormSrgb, 4};
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return {WGPUTextureFormat_RGB10A2Unorm, 4};
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return {WGPUTextureFormat_RG11B10Ufloat, 4};
    case VK_FORMAT_R16_UNORM:              return {WGPUTextureFormat_R16Unorm, 2};
    case VK_FORMAT_R16_SNORM:              return {WGPUTextureFormat_R16Snorm, 2};
    case VK_FORMAT_R16_UINT:               return {WGPUTextureFormat_R16Uint, 2};
    case VK_FORMAT_R16_SINT:               return {WGPUTextureFormat_R16Sint, 2};
    case VK_FORMAT_R16_SFLOAT:             return {WGPUTextureFormat_R16Float, 2};
    case VK_FORMAT_R16G16_UINT:            return {WGPUTextureFormat_RG16Uint, 4};
    case VK_FORMAT_R16G16_SFLOAT:          return {WGPUTextureFormat_RG16Float, 4};
    case VK_FORMAT_R16G16B16A16_SFLOAT:    return {WGPUTextureFormat_RGBA16Float, 8};
    case VK_FORMAT_R16G16B16A16_UINT:      return {WGPUTextureFormat_RGBA16Uint, 8};
    case VK_FORMAT_R32_UINT:               return {WGPUTextureFormat_R32Uint, 4};
    case VK_FORMAT_R32_SINT:               return {WGPUTextureFormat_R32Sint, 4};
    case VK_FORMAT_R32_SFLOAT:             return {WGPUTextureFormat_R32Float, 4};
    case VK_FORMAT_R32G32_UINT:            return {WGPUTextureFormat_RG32Uint, 8};
    case VK_FORMAT_R32G32_SFLOAT:          return {WGPUTextureFormat_RG32Float, 8};
    case VK_FORMAT_R32G32B32A32_UINT:      return {WGPUTextureFormat_RGBA32Uint, 16};
    case VK_FORMAT_R32G32B32A32_SFLOAT:    return {WGPUTextureFormat_RGBA32Float, 16};
    case VK_FORMAT_D16_UNORM:              return {WGPUTextureFormat_Depth16Unorm, 2, 1, true};
    case VK_FORMAT_D32_SFLOAT:             return {WGPUTextureFormat_Depth32Float, 4, 1, true};
    case VK_FORMAT_D24_UNORM_S8_UINT:      return {WGPUTextureFormat_Depth24PlusStencil8, 4, 1, true, true};
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        // Optional in WebGPU. Without it the nearest is 24-bit depth, which
        // is what the renderer falls back to on hardware that lacks it too.
        return {gpu().depth32Stencil8 ? WGPUTextureFormat_Depth32FloatStencil8
                                      : WGPUTextureFormat_Depth24PlusStencil8,
                gpu().depth32Stencil8 ? 8u : 4u, 1, true, true};
    case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
    case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
        return {gpu().bc ? WGPUTextureFormat_BC1RGBAUnorm : WGPUTextureFormat_Undefined, 8, 4};
    case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
    case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
        return {gpu().bc ? WGPUTextureFormat_BC1RGBAUnormSrgb : WGPUTextureFormat_Undefined, 8, 4};
    case VK_FORMAT_BC2_UNORM_BLOCK:
        return {gpu().bc ? WGPUTextureFormat_BC2RGBAUnorm : WGPUTextureFormat_Undefined, 16, 4};
    case VK_FORMAT_BC2_SRGB_BLOCK:
        return {gpu().bc ? WGPUTextureFormat_BC2RGBAUnormSrgb : WGPUTextureFormat_Undefined, 16, 4};
    case VK_FORMAT_BC3_UNORM_BLOCK:
        return {gpu().bc ? WGPUTextureFormat_BC3RGBAUnorm : WGPUTextureFormat_Undefined, 16, 4};
    case VK_FORMAT_BC3_SRGB_BLOCK:
        return {gpu().bc ? WGPUTextureFormat_BC3RGBAUnormSrgb : WGPUTextureFormat_Undefined, 16, 4};
    case VK_FORMAT_BC4_UNORM_BLOCK:
        return {gpu().bc ? WGPUTextureFormat_BC4RUnorm : WGPUTextureFormat_Undefined, 8, 4};
    case VK_FORMAT_BC5_UNORM_BLOCK:
        return {gpu().bc ? WGPUTextureFormat_BC5RGUnorm : WGPUTextureFormat_Undefined, 16, 4};
    case VK_FORMAT_BC7_UNORM_BLOCK:
        return {gpu().bc ? WGPUTextureFormat_BC7RGBAUnorm : WGPUTextureFormat_Undefined, 16, 4};
    default:
        return {};
    }
}

WGPUVertexFormat vertexFormat(VkFormat format) {
    switch (format) {
    case VK_FORMAT_R32_SFLOAT:          return WGPUVertexFormat_Float32;
    case VK_FORMAT_R32G32_SFLOAT:       return WGPUVertexFormat_Float32x2;
    case VK_FORMAT_R32G32B32_SFLOAT:    return WGPUVertexFormat_Float32x3;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return WGPUVertexFormat_Float32x4;
    case VK_FORMAT_R32_UINT:            return WGPUVertexFormat_Uint32;
    case VK_FORMAT_R32G32_UINT:         return WGPUVertexFormat_Uint32x2;
    case VK_FORMAT_R32G32B32_UINT:      return WGPUVertexFormat_Uint32x3;
    case VK_FORMAT_R32G32B32A32_UINT:   return WGPUVertexFormat_Uint32x4;
    case VK_FORMAT_R32_SINT:            return WGPUVertexFormat_Sint32;
    case VK_FORMAT_R32G32_SINT:         return WGPUVertexFormat_Sint32x2;
    case VK_FORMAT_R32G32B32_SINT:      return WGPUVertexFormat_Sint32x3;
    case VK_FORMAT_R32G32B32A32_SINT:   return WGPUVertexFormat_Sint32x4;
    case VK_FORMAT_R16G16_SFLOAT:       return WGPUVertexFormat_Float16x2;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return WGPUVertexFormat_Float16x4;
    case VK_FORMAT_R16G16_UNORM:        return WGPUVertexFormat_Unorm16x2;
    case VK_FORMAT_R16G16B16A16_UNORM:  return WGPUVertexFormat_Unorm16x4;
    case VK_FORMAT_R16G16_SNORM:        return WGPUVertexFormat_Snorm16x2;
    case VK_FORMAT_R16G16B16A16_SNORM:  return WGPUVertexFormat_Snorm16x4;
    case VK_FORMAT_R16G16_UINT:         return WGPUVertexFormat_Uint16x2;
    case VK_FORMAT_R16G16B16A16_UINT:   return WGPUVertexFormat_Uint16x4;
    case VK_FORMAT_R16G16_SINT:         return WGPUVertexFormat_Sint16x2;
    case VK_FORMAT_R16G16B16A16_SINT:   return WGPUVertexFormat_Sint16x4;
    case VK_FORMAT_R8G8_UNORM:          return WGPUVertexFormat_Unorm8x2;
    case VK_FORMAT_R8G8B8A8_UNORM:      return WGPUVertexFormat_Unorm8x4;
    case VK_FORMAT_B8G8R8A8_UNORM:      return WGPUVertexFormat_Unorm8x4BGRA;
    case VK_FORMAT_R8G8_SNORM:          return WGPUVertexFormat_Snorm8x2;
    case VK_FORMAT_R8G8B8A8_SNORM:      return WGPUVertexFormat_Snorm8x4;
    case VK_FORMAT_R8G8_UINT:           return WGPUVertexFormat_Uint8x2;
    case VK_FORMAT_R8G8B8A8_UINT:       return WGPUVertexFormat_Uint8x4;
    case VK_FORMAT_R8G8_SINT:           return WGPUVertexFormat_Sint8x2;
    case VK_FORMAT_R8G8B8A8_SINT:       return WGPUVertexFormat_Sint8x4;
    default:                            return WGPUVertexFormat_Force32;
    }
}

} // namespace wgpuvk
