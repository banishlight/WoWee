// Frame capture: the swapchain image of one frame, printed to the console.
//
// Headless Chromium cannot screenshot a WebGPU canvas, and a canvas read back
// through 2D drawing comes back empty there too, so an automated look at what
// the port draws has to come from WebGPU itself. Opening the page as
// wowee.html?capture=N copies frame N's canvas texture out, encodes it as a PNG
// and logs it as one "WOWEE_CAPTURE <base64>" line, which a test run can pull
// out of the browser log.

#include "wgpu_layer.hpp"

#include <emscripten/console.h>
#include <emscripten/emscripten.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "stb_image_write.h"
#include "core/logger.hpp"

namespace wgpuvk {

namespace {

struct Pending {
    WGPUBuffer buffer;
    uint32_t width, height, rowBytes;
    bool bgra;
};

std::string base64(const std::vector<uint8_t>& in) {
    static const char* t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    for (size_t i = 0; i < in.size(); i += 3) {
        uint32_t v = uint32_t(in[i]) << 16;
        if (i + 1 < in.size()) v |= uint32_t(in[i + 1]) << 8;
        if (i + 2 < in.size()) v |= in[i + 2];
        out += t[(v >> 18) & 63];
        out += t[(v >> 12) & 63];
        out += i + 1 < in.size() ? t[(v >> 6) & 63] : '=';
        out += i + 2 < in.size() ? t[v & 63] : '=';
    }
    return out;
}

void onMapped(WGPUMapAsyncStatus status, WGPUStringView, void* userdata, void*) {
    auto* p = static_cast<Pending*>(userdata);
    if (status == WGPUMapAsyncStatus_Success) {
        const auto* src = static_cast<const uint8_t*>(
            wgpuBufferGetConstMappedRange(p->buffer, 0, size_t(p->rowBytes) * p->height));
        std::vector<uint8_t> rgba(size_t(p->width) * p->height * 4);
        for (uint32_t y = 0; y < p->height; ++y) {
            const uint8_t* row = src + size_t(y) * p->rowBytes;
            uint8_t* dst = rgba.data() + size_t(y) * p->width * 4;
            for (uint32_t x = 0; x < p->width; ++x) {
                dst[x * 4 + 0] = row[x * 4 + (p->bgra ? 2 : 0)];
                dst[x * 4 + 1] = row[x * 4 + 1];
                dst[x * 4 + 2] = row[x * 4 + (p->bgra ? 0 : 2)];
                dst[x * 4 + 3] = 255;
            }
        }
        std::vector<uint8_t> png;
        stbi_write_png_to_func(
            [](void* ctx, void* data, int size) {
                auto* v = static_cast<std::vector<uint8_t>*>(ctx);
                v->insert(v->end(), static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + size);
            },
            &png, int(p->width), int(p->height), 4, rgba.data(), int(p->width) * 4);
        const std::string line = "WOWEE_CAPTURE " + base64(png);
        emscripten_out(line.c_str());
        wgpuBufferUnmap(p->buffer);
    } else {
        LOG_ERROR("WebGPU capture: map failed");
    }
    wgpuBufferRelease(p->buffer);
    delete p;
}

} // namespace

/// Called at present with the frame's canvas texture, before it is let go.
void maybeCapture(WGPUTexture texture, uint32_t width, uint32_t height, bool bgra) {
    static const long wanted = [] {
        const char* v = std::getenv("WOWEE_CAPTURE_FRAME");
        return v ? std::strtol(v, nullptr, 10) : -1L;
    }();
    static long frame = 0;
    if (wanted < 0 || frame++ != wanted || !texture) return;

    const uint32_t rowBytes = (width * 4 + 255) / 256 * 256;
    WGPUBufferDescriptor bd = WGPU_BUFFER_DESCRIPTOR_INIT;
    bd.size = uint64_t(rowBytes) * height;
    bd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    WGPUBuffer buffer = wgpuDeviceCreateBuffer(gpu().device, &bd);

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(gpu().device, nullptr);
    WGPUTexelCopyTextureInfo src = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
    src.texture = texture;
    WGPUTexelCopyBufferInfo dst = WGPU_TEXEL_COPY_BUFFER_INFO_INIT;
    dst.buffer = buffer;
    dst.layout.bytesPerRow = rowBytes;
    dst.layout.rowsPerImage = height;
    WGPUExtent3D extent = {width, height, 1};
    wgpuCommandEncoderCopyTextureToBuffer(enc, &src, &dst, &extent);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(gpu().queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);

    WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowSpontaneous;
    cb.callback = onMapped;
    cb.userdata1 = new Pending{buffer, width, height, rowBytes, bgra};
    wgpuBufferMapAsync(buffer, WGPUMapMode_Read, 0, bd.size, cb);
    LOG_WARNING("WebGPU capture: frame ", wanted, " (", width, "x", height, ") requested");
}

} // namespace wgpuvk
