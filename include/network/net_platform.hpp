#pragma once

// Cross-platform socket abstractions for Windows (Winsock2) and POSIX.

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #ifdef _MSC_VER
    #pragma comment(lib, "ws2_32.lib")
  #endif

  using socket_t  = SOCKET;
#ifndef __MINGW32__
  using ssize_t   = int;             // recv/send return int on MSVC
#endif

  inline constexpr socket_t INVALID_SOCK = INVALID_SOCKET;

#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <netdb.h>
  #include <netinet/tcp.h>
  #include <unistd.h>
  #include <fcntl.h>
  #include <cerrno>

  using socket_t = int;

  inline constexpr socket_t INVALID_SOCK = -1;

#endif

#include <cstring>
#include <string>

#include "core/logger.hpp"

namespace wowee {
namespace net {

// ---- Winsock lifecycle (no-op on Linux) ----

#ifdef _WIN32
struct WinsockInit {
    WinsockInit() {
        WSADATA wsa;
        WSAStartup(MAKEWORD(2, 2), &wsa);
    }
    ~WinsockInit() { WSACleanup(); }
};
// Call once at program start (e.g. as a static in Application).
inline void ensureInit() {
    static WinsockInit instance;
}
#else
inline void ensureInit() {}
#endif

#ifdef __EMSCRIPTEN__
// ---- The browser (src/platform/web_socket.cpp) ----
//
// A browser cannot open a TCP connection, only a WebSocket. Each socket here
// is one, to the relay tools/serve-wasm.py runs beside the page, which opens
// the TCP connection to the game server and passes bytes both ways. The
// handles are small integers like a descriptor, so the socket classes above
// this do not know the difference.
socket_t webOpen(const std::string& host, uint16_t port);
bool webConnect(socket_t s, const std::string& what);
void webClose(socket_t s);
ssize_t webSend(socket_t s, const uint8_t* data, size_t len);
ssize_t webRecv(socket_t s, uint8_t* buf, size_t len);
#endif

// ---- Portable helpers ----

inline void closeSocket(socket_t s) {
#ifdef __EMSCRIPTEN__
    webClose(s);
    return;
#endif
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
}

inline bool setNonBlocking(socket_t s) {
#ifdef __EMSCRIPTEN__
    (void)s;
    return true;   // always, in the browser
#elif defined(_WIN32)
    u_long mode = 1;
    return ioctlsocket(s, FIONBIO, &mode) == 0;
#else
    int flags = fcntl(s, F_GETFL, 0);
    return fcntl(s, F_SETFL, flags | O_NONBLOCK) != -1;
#endif
}

inline int lastError() {
#ifdef _WIN32
    return WSAGetLastError();
#else
    return errno;
#endif
}

inline bool isWouldBlock(int err) {
#ifdef _WIN32
    return err == WSAEWOULDBLOCK;
#else
    return err == EAGAIN || err == EWOULDBLOCK;
#endif
}

// Returns true for errors that mean the peer closed the connection cleanly.
// On Windows, WSAENOTCONN / WSAECONNRESET / WSAESHUTDOWN can be returned by
// recv() when the server closes the connection, rather than returning 0.
inline bool isConnectionClosed(int err) {
#ifdef _WIN32
    return err == WSAENOTCONN    ||  // socket not connected (server closed)
           err == WSAECONNRESET  ||  // connection reset by peer
           err == WSAESHUTDOWN   ||  // socket shut down
           err == WSAECONNABORTED;   // connection aborted
#else
    return err == ENOTCONN || err == ECONNRESET;
#endif
}

inline bool isInProgress(int err) {
#ifdef _WIN32
    return err == WSAEWOULDBLOCK || err == WSAEALREADY;
#else
    return err == EINPROGRESS;
#endif
}

inline const char* errorString(int err) {
#ifdef _WIN32
    // Simple thread-local buffer for FormatMessage
    thread_local char buf[256];
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                   nullptr, err, 0, buf, sizeof(buf), nullptr);
    return buf;
#else
    return strerror(err);
#endif
}

// Portable send - Windows recv/send take char*, not void*.
inline ssize_t portableSend(socket_t s, const uint8_t* data, size_t len) {
#ifdef __EMSCRIPTEN__
    return webSend(s, data, len);
#endif
    return ::send(s, reinterpret_cast<const char*>(data), static_cast<int>(len), 0);
}

inline ssize_t portableRecv(socket_t s, uint8_t* buf, size_t len) {
#ifdef __EMSCRIPTEN__
    return webRecv(s, buf, len);
#endif
    return ::recv(s, reinterpret_cast<char*>(buf), static_cast<int>(len), 0);
}

/// Open a non-blocking TCP socket and resolve host into an address to connect
/// to. Answers INVALID_SOCK when either step fails, having cleaned up after
/// itself.
///
/// The connect itself is deliberately left to the caller: TCPSocket and
/// WorldSocket wait on it differently, and only this setup was identical
/// between them. It was written out twice, down to the log lines.
inline socket_t openResolvedSocket(const std::string& host, uint16_t port,
                                   struct sockaddr_in& addr) {
#ifdef __EMSCRIPTEN__
    // No resolving here: the relay does it, and the browser could not.
    memset(&addr, 0, sizeof(addr));
    return webOpen(host, port);
#endif
    socket_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == INVALID_SOCK) {
        LOG_ERROR("Failed to create socket");
        return INVALID_SOCK;
    }
    setNonBlocking(fd);

    struct addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || res == nullptr) {
        LOG_ERROR("Failed to resolve host: ", host);
        closeSocket(fd);
        return INVALID_SOCK;
    }

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr = reinterpret_cast<struct sockaddr_in*>(res->ai_addr)->sin_addr;
    addr.sin_port = htons(port);
    freeaddrinfo(res);
    return fd;
}

/// Connect a socket from openResolvedSocket, waiting up to timeoutSec for the
/// non-blocking connect to finish, and turn off Nagle's algorithm. On failure
/// the socket is closed and the reason logged; `what` names the peer in it.
///
/// TCPSocket and WorldSocket each wrote this out, differing only in the
/// timeout and the log line.
inline bool connectSocket(socket_t fd, const sockaddr_in& addr, int timeoutSec,
                          const std::string& what) {
#ifdef __EMSCRIPTEN__
    (void)addr;
    (void)timeoutSec;
    return webConnect(fd, what);
#endif
    int result = ::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr));
    if (result < 0) {
        int err = lastError();
        if (!isInProgress(err)) {
            LOG_ERROR("Failed to connect to ", what, ": ", errorString(err));
            closeSocket(fd);
            return false;
        }

        // Wait for the connect to complete. On Windows, calling recv() before
        // it does returns WSAENOTCONN, so writability has to be polled first.
        fd_set writefds, errfds;
        FD_ZERO(&writefds);
        FD_ZERO(&errfds);
        FD_SET(fd, &writefds);
        FD_SET(fd, &errfds);
        struct timeval tv;
        tv.tv_sec = timeoutSec;
        tv.tv_usec = 0;
        if (::select(static_cast<int>(fd) + 1, nullptr, &writefds, &errfds, &tv) <= 0) {
            LOG_ERROR("Connection timed out (", what, ")");
            closeSocket(fd);
            return false;
        }

        // Writable does not mean connected on every platform.
        int sockErr = 0;
        socklen_t errLen = sizeof(sockErr);
        getsockopt(fd, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&sockErr), &errLen);
        if (sockErr != 0) {
            LOG_ERROR("Failed to connect to ", what, ": ", errorString(sockErr));
            closeSocket(fd);
            return false;
        }
    }

    // Send small packets immediately.
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char*>(&one), sizeof(one));
    return true;
}

} // namespace net
} // namespace wowee
