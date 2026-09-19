#pragma once

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>

namespace wowee::platform {

/// The window's size in pixels, which is what a swapchain is built at.
///
/// SDL_Vulkan_GetDrawableSize everywhere but the browser, where SDL has no
/// Vulkan integration and the canvas is drawn at its own pixel size.
inline void drawableSize(SDL_Window* window, int* w, int* h) {
#ifdef __EMSCRIPTEN__
    SDL_GetWindowSize(window, w, h);
#else
    SDL_Vulkan_GetDrawableSize(window, w, h);
#endif
}

} // namespace wowee::platform
