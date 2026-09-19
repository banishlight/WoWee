// Giving the browser its turn in the middle of a call.
//
// A page's main thread belongs to the browser, which draws, reads the network
// and hands out the work other threads queue for it only when the page's code
// has returned. The client's loops do not return: the world load runs for
// seconds, drawing its loading screen as it goes, and in a browser none of it
// was seen - and the packets it waited on never arrived - until it was over.
//
// JSPI (the build's -sJSPI) lets a call into the client suspend on a promise
// and resume later, with its whole stack intact. So the main loop is an
// ordinary loop in main, and presenting a frame waits for the next animation
// frame: the browser shows what was drawn, runs its events, and comes back.

#ifdef __EMSCRIPTEN__

#include "platform/web_frame.hpp"

#include <emscripten/emscripten.h>

namespace wowee::platform {

namespace {

EM_ASYNC_JS(void, waitForAnimationFrame, (), {
    await new Promise(function (resolve) { requestAnimationFrame(resolve); });
});

uint64_t gYielded = 0;

} // namespace

void yieldFrame() {
    waitForAnimationFrame();
    ++gYielded;
}

uint64_t framesYielded() {
    return gYielded;
}

} // namespace wowee::platform

#endif // __EMSCRIPTEN__
