#pragma once

#include <cstdint>

namespace wowee::platform {

/// Hands the page's main thread back to the browser until its next animation
/// frame (see web_frame.cpp). Browser build only.
///
/// The browser shows a frame, delivers network data and runs the work other
/// threads hand the main thread only between calls into the client. A long
/// step - the world load and its loading screen - calls this wherever it
/// presents a frame, through Window::swapBuffers, and carries on from there
/// when the browser comes back.
void yieldFrame();

/// How many times yieldFrame has returned, so the main loop can tell whether
/// a frame already gave the browser its turn.
uint64_t framesYielded();

} // namespace wowee::platform
