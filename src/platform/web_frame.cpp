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
    // The next animation frame, or 100ms, whichever comes first.
    //
    // A browser stops calling back for frames whenever it is not painting the
    // page - a hidden tab, a window behind another, a compositor that has
    // given up on the canvas. Waiting only for the callback means the client
    // stops with it: no network, no keepalives, and the game server drops the
    // connection. Firefox stops calling back mid-load, which is what froze it
    // there; Chromium kept going, so this never showed until now.
    //
    // So the frame waits for the callback, and carries on without it if it
    // does not come. The picture is only as fresh as the browser's painting
    // either way, but the client stays alive behind it.
    await new Promise(function (resolve) {
        var done = false;
        var frame = requestAnimationFrame(function () {
            if (done) return;
            done = true;
            resolve();
        });
        setTimeout(function () {
            if (done) return;
            done = true;
            cancelAnimationFrame(frame);
            resolve();
        }, 100);
    });
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
