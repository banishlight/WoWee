// Sockets in the browser: a WebSocket each, to the relay beside the page.
//
// The socket classes read and write through net::portableRecv/portableSend
// from whichever thread they like - the world socket has a pump thread of its
// own. A WebSocket belongs to the page's main thread, and its data arrives
// there, in callbacks that only run when that thread returns to the browser.
// So each socket here is a buffer between the two: callbacks on the main thread
// fill it, recv drains it from any thread, and sends are handed to the main
// thread without waiting (a pump thread blocked on a main thread that is
// joining it would never return).
//
// A connect cannot wait for the WebSocket to open either - the main thread is
// the one waiting. It answers at once; bytes sent before the socket opens are
// held and go out when it does, and a socket that fails to open reads as a
// closed connection, which the client already handles.

#ifdef __EMSCRIPTEN__

#include "network/net_platform.hpp"

#include <emscripten/emscripten.h>
#include <emscripten/proxying.h>
#include <emscripten/threading.h>
#include <emscripten/websocket.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace wowee::net {

namespace {

enum class State { Idle, Connecting, Open, Closed };

struct WebSock {
    std::mutex mutex;
    std::string target;                 // host:port for the relay
    State state = State::Idle;
    EMSCRIPTEN_WEBSOCKET_T ws = 0;
    std::deque<uint8_t> in;             // received, not yet read
    std::vector<uint8_t> held;          // sent before the socket opened
};

std::mutex gTableMutex;
std::unordered_map<socket_t, std::shared_ptr<WebSock>> gTable;
socket_t gNext = 1000;

std::shared_ptr<WebSock> find(socket_t s) {
    std::lock_guard<std::mutex> lock(gTableMutex);
    auto it = gTable.find(s);
    return it == gTable.end() ? nullptr : it->second;
}

/// ws://<page host>/relay?target=host:port, on the page's own origin.
EM_JS(char*, relayUrl, (const char* target), {
    var scheme = location.protocol === 'https:' ? 'wss://' : 'ws://';
    return stringToNewUTF8(scheme + location.host + '/relay?target=' +
                           encodeURIComponent(UTF8ToString(target)));
});

/// Runs fn on the main thread without waiting for it.
template <typename F>
void onMain(F&& fn) {
    if (emscripten_is_main_runtime_thread()) {
        fn();
        return;
    }
    auto* task = new std::function<void()>(std::forward<F>(fn));
    emscripten_proxy_async(emscripten_proxy_get_system_queue(), emscripten_main_runtime_thread_id(),
                           [](void* arg) {
                               auto* t = static_cast<std::function<void()>*>(arg);
                               (*t)();
                               delete t;
                           },
                           task);
}

bool onOpen(int, const EmscriptenWebSocketOpenEvent*, void* user) {
    auto* sock = static_cast<WebSock*>(user);
    std::lock_guard<std::mutex> lock(sock->mutex);
    sock->state = State::Open;
    if (!sock->held.empty()) {
        emscripten_websocket_send_binary(sock->ws, sock->held.data(),
                                         static_cast<uint32_t>(sock->held.size()));
        sock->held.clear();
    }
    return true;
}

bool onMessage(int, const EmscriptenWebSocketMessageEvent* e, void* user) {
    auto* sock = static_cast<WebSock*>(user);
    std::lock_guard<std::mutex> lock(sock->mutex);
    sock->in.insert(sock->in.end(), e->data, e->data + e->numBytes);
    return true;
}

bool onClose(int, const EmscriptenWebSocketCloseEvent* e, void* user) {
    auto* sock = static_cast<WebSock*>(user);
    std::lock_guard<std::mutex> lock(sock->mutex);
    if (sock->state == State::Connecting)
        LOG_ERROR("Relay connection to ", sock->target, " failed (code ", e->code,
                  ") - is tools/serve-wasm.py serving the page, and the server up?");
    sock->state = State::Closed;
    return true;
}

bool onError(int, const EmscriptenWebSocketErrorEvent*, void* user) {
    auto* sock = static_cast<WebSock*>(user);
    std::lock_guard<std::mutex> lock(sock->mutex);
    sock->state = State::Closed;
    return true;
}

} // namespace

socket_t webOpen(const std::string& host, uint16_t port) {
    auto sock = std::make_shared<WebSock>();
    sock->target = host + ":" + std::to_string(port);
    std::lock_guard<std::mutex> lock(gTableMutex);
    const socket_t fd = gNext++;
    gTable.emplace(fd, std::move(sock));
    return fd;
}

bool webConnect(socket_t s, const std::string& what) {
    auto sock = find(s);
    if (!sock) return false;
    {
        std::lock_guard<std::mutex> lock(sock->mutex);
        sock->state = State::Connecting;
    }
    onMain([sock, what] {
        char* url = relayUrl(sock->target.c_str());
        EmscriptenWebSocketCreateAttributes attrs;
        emscripten_websocket_init_create_attributes(&attrs);
        attrs.url = url;
        attrs.protocols = nullptr;
        attrs.createOnMainThread = true;
        const EMSCRIPTEN_WEBSOCKET_T ws = emscripten_websocket_new(&attrs);
        std::free(url);
        std::lock_guard<std::mutex> lock(sock->mutex);
        if (ws <= 0) {
            LOG_ERROR("Could not open a WebSocket for ", what);
            sock->state = State::Closed;
            return;
        }
        sock->ws = ws;
        // The callbacks hold a raw pointer; the table entry is dropped on
        // close only after the WebSocket is deleted, so it outlives them.
        void* user = sock.get();
        const pthread_t main = emscripten_main_runtime_thread_id();
        emscripten_websocket_set_onopen_callback_on_thread(ws, user, onOpen, main);
        emscripten_websocket_set_onmessage_callback_on_thread(ws, user, onMessage, main);
        emscripten_websocket_set_onclose_callback_on_thread(ws, user, onClose, main);
        emscripten_websocket_set_onerror_callback_on_thread(ws, user, onError, main);
    });
    LOG_INFO("Connecting to ", what, " through the page's relay");
    return true;
}

void webClose(socket_t s) {
    std::shared_ptr<WebSock> sock;
    {
        std::lock_guard<std::mutex> lock(gTableMutex);
        auto it = gTable.find(s);
        if (it == gTable.end()) return;
        sock = std::move(it->second);
        gTable.erase(it);
    }
    // The shared_ptr rides along, so the object lives until the WebSocket
    // and its callbacks are gone.
    onMain([sock] {
        std::lock_guard<std::mutex> lock(sock->mutex);
        if (sock->ws > 0) {
            emscripten_websocket_close(sock->ws, 1000, "closed");
            emscripten_websocket_delete(sock->ws);
            sock->ws = 0;
        }
        sock->state = State::Closed;
    });
}

ssize_t webSend(socket_t s, const uint8_t* data, size_t len) {
    auto sock = find(s);
    if (!sock) { errno = EBADF; return -1; }
    {
        std::lock_guard<std::mutex> lock(sock->mutex);
        if (sock->state == State::Closed) { errno = ECONNRESET; return -1; }
    }
    std::vector<uint8_t> bytes(data, data + len);
    onMain([sock, bytes = std::move(bytes)]() mutable {
        std::lock_guard<std::mutex> lock(sock->mutex);
        if (sock->state == State::Open) {
            emscripten_websocket_send_binary(sock->ws, bytes.data(), static_cast<uint32_t>(bytes.size()));
        } else if (sock->state == State::Connecting) {
            sock->held.insert(sock->held.end(), bytes.begin(), bytes.end());
        }
    });
    return static_cast<ssize_t>(len);
}

ssize_t webRecv(socket_t s, uint8_t* buf, size_t len) {
    auto sock = find(s);
    if (!sock) { errno = EBADF; return -1; }
    std::lock_guard<std::mutex> lock(sock->mutex);
    if (!sock->in.empty()) {
        const size_t n = std::min(len, sock->in.size());
        std::copy(sock->in.begin(), sock->in.begin() + static_cast<std::ptrdiff_t>(n), buf);
        sock->in.erase(sock->in.begin(), sock->in.begin() + static_cast<std::ptrdiff_t>(n));
        return static_cast<ssize_t>(n);
    }
    if (sock->state == State::Closed) return 0;   // the peer is gone
    errno = EWOULDBLOCK;
    return -1;
}

} // namespace wowee::net

#endif // __EMSCRIPTEN__
