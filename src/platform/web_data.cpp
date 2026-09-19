// The game data, in the browser.
//
// The client reads ./Data with ordinary file calls from all over - the asset
// manager, fonts, the interface loader. Several gigabytes of it cannot be
// packed into the page, so /Data is a WasmFS directory whose files are fetched
// from the server the first time each one is read, and only then.
//
// WasmFS only serves files it has been told exist, so the server provides an
// index of everything under Data/ (tools/serve-wasm.py writes it on request),
// and every entry is created up front. Creating one fetches nothing; a read or
// a stat does.

#ifdef __EMSCRIPTEN__

#include "platform/web_data.hpp"

#include <emscripten/wasmfs.h>
#include <emscripten/threading.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_set>

#include "core/logger.hpp"

namespace wowee::platform {

namespace {

constexpr const char* kMount = "/Data";
constexpr const char* kIndex = "/Data/.wowee-index";

void makeParents(const std::string& path, std::unordered_set<std::string>& made) {
    for (size_t slash = path.find('/', 1); slash != std::string::npos; slash = path.find('/', slash + 1)) {
        std::string dir = path.substr(0, slash);
        if (made.insert(dir).second) mkdir(dir.c_str(), 0777);
    }
}

bool touch(const char* path) {
    int fd = open(path, O_CREAT | O_RDONLY, 0444);
    if (fd < 0) return false;
    close(fd);
    return true;
}

} // namespace

bool mountWebData() {
    // Must not run on the browser's main thread: the backend's worker is
    // spawned synchronously, which needs the main thread free to do it.
    if (emscripten_is_main_browser_thread()) {
        LOG_ERROR("mountWebData called on the main thread");
        return false;
    }
    // A chunk bigger than any game file, so every file arrives in one piece.
    // With the default 16 MB chunks, a file over 32 MB from a server without
    // range requests (Python's http.server) came back with everything past
    // the first chunk wrong - the 32 MB asset manifest among them.
    backend_t fetch = wasmfs_create_fetch_backend("Data", 1u << 30);
    if (wasmfs_create_directory(kMount, 0777, fetch) != 0) {
        LOG_ERROR("Could not mount ", kMount);
        return false;
    }
    if (!touch(kIndex)) {
        LOG_ERROR("Could not create ", kIndex);
        return false;
    }
    std::ifstream in(kIndex);
    if (!in) {
        LOG_ERROR("No game data index from the server (", kIndex, ")");
        return false;
    }
    std::unordered_set<std::string> made{kMount};
    std::string line;
    size_t files = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const std::string path = std::string(kMount) + "/" + line;
        makeParents(path, made);
        if (touch(path.c_str())) ++files;
    }
    LOG_WARNING("Game data: ", files, " files under ", kMount, ", fetched on first read");
    return files > 0;
}

} // namespace wowee::platform

#endif // __EMSCRIPTEN__
