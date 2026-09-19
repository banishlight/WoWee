#include "core/memory_monitor.hpp"
#include "core/logger.hpp"
#include "core/byte_size.hpp"
#include <fstream>
#include <string>
#include <sstream>
#ifdef _WIN32
#include <windows.h>
#elif defined(__EMSCRIPTEN__)
#include <emscripten/heap.h>
#include <malloc.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#else
#include <sys/sysinfo.h>
#endif

namespace wowee {
namespace core {

#if !defined(_WIN32) && !defined(__APPLE__) && !defined(__EMSCRIPTEN__)
namespace {
size_t readMemAvailableBytesFromProc() {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) return 0;

    std::string line;
    while (std::getline(meminfo, line)) {
        // /proc/meminfo format: "MemAvailable:  123456789 kB"
        static constexpr size_t kFieldPrefixLen = 13; // strlen("MemAvailable:")
        if (line.rfind("MemAvailable:", 0) != 0) continue;
        std::istringstream iss(line.substr(kFieldPrefixLen));
        size_t kb = 0;
        iss >> kb;
        if (kb > 0) return kb * 1024ull;
        break;
    }
    return 0;
}
} // namespace
#endif // !_WIN32 && !__APPLE__ && !__EMSCRIPTEN__

MemoryMonitor& MemoryMonitor::getInstance() {
    static MemoryMonitor instance;
    return instance;
}

void MemoryMonitor::initialize() {
    constexpr size_t kOneGB = 1024ull * 1024 * 1024;
    // Fallback if OS API unavailable - 16 GB is a safe conservative estimate
    // that prevents over-aggressive asset caching on unknown hardware.
    constexpr size_t kFallbackRAM = mbToBytes(16 * 1024);

#ifdef _WIN32
    ULONGLONG totalKB = 0;
    if (GetPhysicallyInstalledSystemMemory(&totalKB)) {
        totalRAM_ = static_cast<size_t>(totalKB) * 1024ull;
        LOG_INFO("System RAM detected: ", totalRAM_ / kOneGB, " GB");
    } else {
        totalRAM_ = kFallbackRAM;
        LOG_WARNING("Could not detect system RAM, assuming 16GB");
    }
#elif defined(__EMSCRIPTEN__)
    // The machine's RAM is not ours to size caches against. Everything the
    // client allocates lives in one wasm heap, which grows up to a limit set at
    // link time (-sMAXIMUM_MEMORY) and can never exceed 4 GB on wasm32.
    totalRAM_ = emscripten_get_heap_max();
    LOG_INFO("Wasm heap limit: ", totalRAM_ / (1024 * 1024), " MB");
#elif defined(__APPLE__)
    int64_t physmem = 0;
    size_t len = sizeof(physmem);
    if (sysctlbyname("hw.memsize", &physmem, &len, nullptr, 0) == 0) {
        totalRAM_ = static_cast<size_t>(physmem);
        LOG_INFO("System RAM detected: ", totalRAM_ / kOneGB, " GB");
    } else {
        totalRAM_ = kFallbackRAM;
        LOG_WARNING("Could not detect system RAM, assuming 16GB");
    }
#else
    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        totalRAM_ = static_cast<size_t>(info.totalram) * info.mem_unit;
        LOG_INFO("System RAM detected: ", totalRAM_ / kOneGB, " GB");
    } else {
        totalRAM_ = kFallbackRAM;
        LOG_WARNING("Could not detect system RAM, assuming 16GB");
    }
#endif
}

size_t MemoryMonitor::getAvailableRAM() const {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<size_t>(status.ullAvailPhys);
    }
    return totalRAM_ / 2;
#elif defined(__EMSCRIPTEN__)
    // What the heap can still grow by. Freed blocks inside it are not counted,
    // so this reads low once the client has been at its peak - which is the
    // safe direction, and it is the only cheap measure there is.
    //
    // Not mallinfo(): it walks every block in the heap, holding the one
    // allocator lock the whole time. At a 2 GB heap that is seconds, with
    // every thread that allocates stopped behind it, which is a freeze.
    const size_t heap = emscripten_get_heap_size();
    return heap < totalRAM_ ? totalRAM_ - heap : 0;
#elif defined(__APPLE__)
    // hw.usermem is a 32-bit kernel sysctl on macOS: on systems with ≥16 GB RAM
    // the value overflows signed int32, truncating to ~2 GB and causing false
    // severe-pressure reports.  Use the Mach VM statistics API instead, which
    // is the same source that Activity Monitor and `vm_stat` use.
    {
        mach_port_t host_port = mach_host_self();
        vm_size_t page_size = 0;
        host_page_size(host_port, &page_size);
        if (page_size > 0) {
            mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
            vm_statistics64_data_t vm_stat;
            if (host_statistics64(host_port, HOST_VM_INFO64,
                                  reinterpret_cast<host_info64_t>(&vm_stat),
                                  &count) == KERN_SUCCESS) {
                // free + inactive pages are reclaimable on demand
                return (static_cast<size_t>(vm_stat.free_count) +
                        static_cast<size_t>(vm_stat.inactive_count)) *
                       static_cast<size_t>(page_size);
            }
        }
    }
    return totalRAM_ / 2;
#else
    // Best source on Linux for reclaimable memory headroom.
    if (size_t memAvailable = readMemAvailableBytesFromProc(); memAvailable > 0) {
        return memAvailable;
    }

    struct sysinfo info;
    if (sysinfo(&info) == 0) {
        // Fallback approximation if /proc/meminfo is unavailable.
        size_t freeBytes = static_cast<size_t>(info.freeram) * info.mem_unit;
        size_t bufferBytes = static_cast<size_t>(info.bufferram) * info.mem_unit;
        size_t available = freeBytes + bufferBytes;
        return (totalRAM_ > 0 && available > totalRAM_) ? totalRAM_ : available;
    }
    return totalRAM_ / 2;  // Fallback: assume 50% available
#endif
}

size_t MemoryMonitor::getRecommendedCacheBudget() const {
    size_t available = getAvailableRAM();
    // Use 50% of available RAM for caches, hard-capped at 16 GB.
    static constexpr size_t kHardCapBytes = mbToBytes(16 * 1024);  // 16 GB
    size_t budget = available / 2;
    return budget < kHardCapBytes ? budget : kHardCapBytes;
}

bool MemoryMonitor::isMemoryPressure() const {
    size_t available = getAvailableRAM();
    // Memory pressure if < 10% RAM available
    return available < (totalRAM_ / 100 * 10);
}

bool MemoryMonitor::isSevereMemoryPressure() const {
    size_t available = getAvailableRAM();
    // Severe pressure if < 15% RAM available - background workers should
    // pause entirely to avoid OOM-killing other applications.
    return available < (totalRAM_ / 100 * 15);
}

} // namespace core
} // namespace wowee
