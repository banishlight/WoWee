// The game data, in the browser.
//
// The client reads ./Data with ordinary file calls from all over - the asset
// manager, fonts, the interface loader. Several gigabytes of it cannot be
// packed into the page, so /Data is a file system of our own whose files are
// downloaded from the server when they are read.
//
// The server lists what is there, with sizes (tools/serve-wasm.py writes the
// list on request), and every entry is created up front. So a stat, a size or
// an existence check never touches the network; a read downloads the whole
// file once, on the thread that reads it, and the copy is dropped when the
// last handle to it closes - the asset manager keeps its own caches.
//
// Emscripten's fetch backend did this before, and was the slowest part of
// entering the world: every file cost two requests, made one after another by
// a single worker whose requests were relayed through the main thread - which
// is busy during a load, so each one waited on a 50 ms timer instead.
// Downloading on the reading thread itself needs no relay, and the terrain
// workers download side by side.
//
// A backend is written against WasmFS's own headers (system/lib/wasmfs), which
// are not a public interface: an emsdk update may need this file adjusted.

#ifdef __EMSCRIPTEN__

#include "platform/web_data.hpp"

#include <emscripten/emscripten.h>
#include <emscripten/wasmfs.h>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// WasmFS internals.
#include "backend.h"
#include "file.h"
#include "memory_backend.h"
#include "wasmfs.h"

#include "core/logger.hpp"

namespace wowee::platform {

namespace {

constexpr const char* kMount = "/Data";
constexpr const char* kIndexUrl = "Data/.wowee-index";

/// Every file under Data/ and its size, from the server's index. Written once
/// before the mount is populated, only read after.
std::unordered_map<std::string, uint32_t> gSizes;

/// GETs url into dst, synchronously, on the calling thread. Returns the size
/// the server sent (only the first `size` bytes are copied), or -1.
///
/// A worker can ask for an ArrayBuffer; the page's own thread may only make a
/// synchronous request for text, so there the bytes come back one per
/// character.
EM_JS(int, fetchInto, (const char* path, uint8_t* dst, int size), {
    var url = UTF8ToString(path).split('/').map(encodeURIComponent).join('/');
    var xhr = new XMLHttpRequest();
    xhr.open('GET', url, false);
    var worker = typeof window == 'undefined';
    if (worker) xhr.responseType = 'arraybuffer';
    else xhr.overrideMimeType('text/plain; charset=x-user-defined');
    try {
        xhr.send();
    } catch (e) {
        return -1;
    }
    if (xhr.status != 200) return -1;
    if (worker) {
        var bytes = new Uint8Array(xhr.response);
        HEAPU8.set(bytes.subarray(0, Math.min(bytes.length, size)), dst);
        return bytes.length;
    }
    var text = xhr.responseText;
    var n = Math.min(text.length, size);
    for (var i = 0; i < n; i++) HEAPU8[dst + i] = text.charCodeAt(i) & 0xff;
    return text.length;
});

/// The index, which has no size to go by: asks for it once to learn how
/// large it is, then again into a buffer that size.
EM_JS(int, fetchIndexSize, (const char* path), {
    var xhr = new XMLHttpRequest();
    xhr.open('HEAD', UTF8ToString(path), false);
    try {
        xhr.send();
    } catch (e) {
        return -1;
    }
    if (xhr.status != 200) return -1;
    return parseInt(xhr.getResponseHeader('Content-Length') || '-1', 10);
});

class WebDataFile : public wasmfs::DataFile {
public:
    WebDataFile(mode_t mode, wasmfs::backend_t backend, std::string path)
        : DataFile(mode, backend), path_(std::move(path)) {
        // The index is relative to Data/.
        const size_t prefix = std::strlen(kMount) + 1;
        if (path_.size() > prefix) {
            auto it = gSizes.find(path_.substr(prefix));
            if (it != gSizes.end()) size_ = it->second;
        }
    }

private:
    int open(wasmfs::oflags_t) override {
        ++opens_;
        return 0;
    }

    int close() override {
        if (opens_ > 0 && --opens_ == 0) {
            data_ = {};
            loaded_ = false;
        }
        return 0;
    }

    ssize_t read(uint8_t* buf, size_t len, off_t offset) override {
        if (!loaded_) {
            data_.resize(size_);
            const int got = fetchInto(path_.c_str() + 1, data_.data(), static_cast<int>(size_));
            if (got < 0) {
                LOG_ERROR("Game data: could not download ", path_);
                data_ = {};
                return -EIO;
            }
            if (static_cast<uint32_t>(got) != size_)
                LOG_WARNING("Game data: ", path_, " is ", got, " bytes, the index says ", size_,
                            " - the index is out of date");
            data_.resize(std::min<size_t>(static_cast<size_t>(got), size_));
            loaded_ = true;
        }
        if (offset < 0 || static_cast<size_t>(offset) >= data_.size()) return 0;
        const size_t n = std::min(len, data_.size() - static_cast<size_t>(offset));
        std::memcpy(buf, data_.data() + offset, n);
        return static_cast<ssize_t>(n);
    }

    ssize_t write(const uint8_t*, size_t, off_t) override { return -EROFS; }
    int setSize(off_t) override { return -EROFS; }
    int flush() override { return 0; }
    off_t getSize() override { return size_; }

    std::string path_;     // "/Data/..." - also the URL, relative to the page
    uint32_t size_ = 0;
    int opens_ = 0;
    bool loaded_ = false;
    std::vector<uint8_t> data_;
};

class WebDataDirectory : public wasmfs::MemoryDirectory {
public:
    WebDataDirectory(mode_t mode, wasmfs::backend_t backend, std::string path)
        : MemoryDirectory(mode, backend), path_(std::move(path)) {}

private:
    std::shared_ptr<wasmfs::DataFile> insertDataFile(const std::string& name, mode_t mode) override {
        auto child = std::make_shared<WebDataFile>(mode, getBackend(), path_ + "/" + name);
        insertChild(name, child);
        return child;
    }

    std::shared_ptr<wasmfs::Directory> insertDirectory(const std::string& name, mode_t mode) override {
        auto child = std::make_shared<WebDataDirectory>(mode, getBackend(), path_ + "/" + name);
        insertChild(name, child);
        return child;
    }

    std::string path_;
};

class WebDataBackend : public wasmfs::Backend {
public:
    std::shared_ptr<wasmfs::DataFile> createFile(mode_t mode) override {
        return std::make_shared<WebDataFile>(mode, this, "");
    }
    std::shared_ptr<wasmfs::Directory> createDirectory(mode_t mode) override {
        return std::make_shared<WebDataDirectory>(mode, this, kMount);
    }
    std::shared_ptr<wasmfs::Symlink> createSymlink(std::string) override {
        return nullptr;
    }
};

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
    // The index: a relative path and a size a line, tab between them.
    const int indexSize = fetchIndexSize(kIndexUrl);
    if (indexSize <= 0) {
        LOG_ERROR("No game data index from the server (", kIndexUrl, ")");
        return false;
    }
    std::string index(static_cast<size_t>(indexSize), '\0');
    const int got = fetchInto(kIndexUrl, reinterpret_cast<uint8_t*>(index.data()), indexSize);
    if (got < 0) {
        LOG_ERROR("Could not download ", kIndexUrl);
        return false;
    }
    index.resize(std::min(got, indexSize));

    std::istringstream in(index);
    std::string line;
    while (std::getline(in, line)) {
        const size_t tab = line.rfind('\t');
        if (tab == std::string::npos) continue;
        gSizes.emplace(line.substr(0, tab), static_cast<uint32_t>(std::strtoul(line.c_str() + tab + 1, nullptr, 10)));
    }

    // The public API's backend_t is the same pointer under a C name.
    auto* backend = reinterpret_cast<::backend_t>(wasmfs::wasmFS.addBackend(std::make_unique<WebDataBackend>()));
    if (wasmfs_create_directory(kMount, 0777, backend) != 0) {
        LOG_ERROR("Could not mount ", kMount);
        return false;
    }
    std::unordered_set<std::string> made{kMount};
    size_t files = 0;
    for (const auto& [rel, size] : gSizes) {
        const std::string path = std::string(kMount) + "/" + rel;
        makeParents(path, made);
        if (touch(path.c_str())) ++files;
    }
    LOG_WARNING("Game data: ", files, " files under ", kMount, ", downloaded as they are read");
    return files > 0;
}

} // namespace wowee::platform

#endif // __EMSCRIPTEN__
