# The browser port

The client compiled to WebAssembly, running in a browser on WebGPU. Branch
`wasm-port`. This is where it stands, how it works, how to test it, and what is
left.

## Where it stands (2026-09-19)

Working, against a real AzerothCore server (3.3.5a):

- Login, realm list, character select with the character model, and entering
  the world. Eversong Woods renders: terrain, WMO buildings, M2 doodads,
  creatures and players with animation, sky, minimap, the interface.
- The frame rate is playable on a desktop GPU in Chromium.
- Entering the world shows the loading screen with its progress, and takes
  about 10 s even in headless Chromium drawing on the CPU.

Not working or not right yet - see *Known problems*:

- Some textures and lighting look off.
- Particles are single pixels.
- Firefox: the device request fails ("Not enough memory left"). Chromium only
  (137 or later, for JSPI).

## How it works

The renderer is unchanged. It still calls Vulkan; the browser build links a
translation layer that implements the ~115 Vulkan functions the renderer,
vk-bootstrap, VMA and ImGui use on top of WebGPU. Everything that is
browser-specific is gated on `__EMSCRIPTEN__` / `EMSCRIPTEN`, and the native
builds are unaffected.

| Piece | Where | What it does |
|---|---|---|
| Vulkan on WebGPU | `src/platform/webgpu/` | The layer. `wgpu_layer.hpp` has the full design notes. |
| Shader translation | `tools/wasm_shaders.py` | GLSL -> SPIR-V -> WGSL at build time (glslc, spirv-opt, naga), plus a manifest keyed by the hash of the SPIR-V the renderer loads. |
| Game data | `src/platform/web_data.cpp` | `/Data` is a WasmFS backend of our own: the server's index gives every file and its size, and a file is downloaded when it is read, on the thread reading it. |
| Main loop | `src/platform/web_frame.cpp` | The native loop in `main`. Presenting a frame suspends it (JSPI) until the browser's next animation frame. |
| Networking | `src/platform/web_socket.cpp` | Sockets are WebSockets to a relay, which opens the TCP connection to the game server. |
| Dev server + relay | `tools/serve-wasm.py` | Serves the page with the isolation headers, the data index, and the WebSocket relay. |
| Device request | `src/platform/webgpu/webgpu_pre.js` | Gets the WebGPU device before `main`, caps the reported core count at 8, maps URL flags to environment variables. |

The layer, in brief:

- **Objects** are plain structs; handles are pointers to them
  (`VK_USE_64_BIT_PTR_DEFINES=1`). WebGPU objects are created lazily on the
  main thread - they are JS objects and exist only there.
- **Command buffers** record into a CPU-side list on any thread;
  `vkQueueSubmit` replays them into a WebGPU encoder on the main thread
  (`replay.cpp`).
- **Host-visible memory** is a CPU copy. Buffers up to 4 MB are uploaded whole
  when a submission uses them. Larger buffers the GPU draws from only get
  non-coherent memory, and only their flushed ranges are uploaded - so the
  renderer flushes what it writes to its mega buffers (`vmaFlushAllocation`,
  a no-op on desktop drivers).
- **Bindings**: WGSL has no combined image sampler, so Vulkan binding `b` is
  WGSL binding `2b` and its sampler `2b + 1`. Push constants are a uniform
  buffer at group 0, binding 900, with a dynamic offset. The shader tool and
  the layer have to agree on this.
- **Sync**: barriers and layout transitions are nothing; fences signal at
  submit.
- **The main loop** is the native one. A page's main thread has to return to
  the browser for anything to be shown, for network data to arrive, and for
  the work other threads hand it to run. With JSPI (`-sJSPI`) a call can
  suspend on a promise and resume with its stack intact, so
  `Window::swapBuffers` - a no-op natively - waits for the next animation
  frame. Long steps such as the world load, which present their own frames,
  give the browser its turn with no change to them. `SDL_Delay` is kept
  synchronous: worker threads call it and cannot suspend.
- **Game data** is read with synchronous requests on the reading thread
  (allowed in workers; text-only on the main thread, which is decoded). Sizes
  come from the index, so a stat or an existence check costs nothing. A file's
  bytes are kept only while it is open. The backend uses WasmFS's internal
  headers, so an emsdk update may need `web_data.cpp` adjusted.
- **Threads**: every thread is a Web Worker from a pool of 32 made before
  `main`. Running out used to deadlock the main thread silently; it now throws
  (`-sPTHREAD_POOL_SIZE_STRICT=2`). Work that could start a thread per item
  (model loads, normal maps) goes through fixed pools in
  `core/thread_pool.hpp`.

## Building and running

The build steps are in `BUILD_INSTRUCTIONS.md` under *WebAssembly*. In short:

```bash
source ~/emsdk/emsdk_env.sh
emcmake cmake -S . -B build-wasm -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DWOWEE_BUILD_TESTS=OFF -DWOWEE_WARNINGS_AS_ERRORS=OFF \
    -DOPENSSL_ROOT_DIR=$PWD/build-wasm-deps/wasm32
cmake --build build-wasm --target wowee
tools/serve-wasm.py                      # http://localhost:8080/wowee.html
```

The extracted game data has to be in `Data/` (`extract_assets.sh`); the build
links it beside the page and the dev server serves it.

Chromium on a Wayland desktop needs X11 mode for Vulkan:

```bash
chromium --ozone-platform=x11 --enable-unsafe-webgpu --enable-features=Vulkan \
    http://localhost:8080/wowee.html
```

The game takes F12, so open DevTools on a blank tab before loading the page.

## Testing without a person at the keyboard

Page parameters, all debugging aids:

| Parameter | Effect |
|---|---|
| `?autologin=user:pass@host[:port]` | Fills the login card and logs in once. |
| `?enterworld` | Enters the world with the last-played character. |
| `?capture=N` | Logs frame N as a base64 PNG on a `WOWEE_CAPTURE` console line. |
| `?offscreen` | Renders to textures instead of the canvas. Needed headless: headless Chromium loses the device on the first canvas draw. |
| `?eager` | Builds pipelines when they are created rather than at first draw, so refused ones show at startup. |
| `?log=debug\|info\|warn` | The log level. The load's timeline is in the info log. |
| `?env=NAME=VALUE,...` | Settings the client reads from the environment, such as the cache budgets. |

A full headless run into the world and a frame of it:

```bash
chromium --headless=new --enable-unsafe-webgpu --enable-logging=stderr \
  'http://127.0.0.1:8080/wowee.html?offscreen&autologin=USER:PASS@HOST&enterworld&capture=1200' \
  2>&1 | tee run.log | grep -o 'WOWEE_CAPTURE [A-Za-z0-9+/=]*' | cut -d' ' -f2 | base64 -d > frame.png
```

Headless Chromium draws on the CPU (SwiftShader), so it is far slower than a
real GPU; use it for correctness, not for frame rates. It cannot reach a real
GPU at all (its Vulkan instance fails), so anything GPU-dependent has to be
run in a window.

A note from a day lost to it: nothing in the browser build may call
`mallinfo()`. It walks every block in the heap holding the allocator lock, so
at a 2 GB heap every thread stops for seconds. It looks exactly like a hang
in the browser - the main thread spins inside the engine, where neither the
debugger nor the profiler can see it.

The layer logs a pipeline or shader WebGPU refuses once, with the reason
(`WebGPU pipeline ... refused`, `WebGPU shader ... does not compile`), and
skips draws with a refused pipeline rather than losing the frame.

## Known problems

1. **Memory.** A page gets 4 GB, and the client is at about 2.4 GB of it once
   the world is loaded - the caches size themselves for a desktop's RAM
   (`memory_monitor.cpp`, and the `WOWEE_*_CACHE_MB` settings). Nothing has
   been tuned for the browser yet, and running out would be a hard failure.
2. **The first frames in the world may stall** while WebGPU builds the
   pipelines they use - pipelines are built at first draw. Headless (CPU)
   runs show single frames of seconds there, which may be SwiftShader alone;
   not yet measured on a GPU. If it shows on one: create them with
   `createRenderPipelineAsync` and skip draws until they are ready, or build
   them during the loading screen.
3. **Textures and lighting partly wrong.** Not yet separated into port bugs
   and client bugs. Compare the same spot in the native client and the
   browser. Suspects on the port side: depth textures read through a nearest
   sampler (`VkSampler_T::ensureNearest`), the `textureQueryLod` stand-in in
   `wasm_shaders.py`, canvas sRGB handling.
4. **Particles are one pixel.** WebGPU has no point size. The shader tool
   already routes `gl_PointSize`/`gl_PointCoord` through varyings (locations
   14/15); what is left is expanding each point to a quad (wrap `main`,
   instanced draw of 6 vertices per point).
5. **No GPU-to-CPU readback.** `vkCmdCopyImageToBuffer` and copies into
   host-visible buffers log a warning and do nothing. Needs `mapAsync` and
   copying into the CPU copy when it completes.
6. **Some compute passes cannot run**: FSR2 and HiZ use read-write storage
   texture formats WebGPU does not allow (rgba16float, rg16float).
7. **Firefox** refuses the device. The fallback chain in `webgpu_pre.js` did
   not help; not investigated further.
8. **Warden** cannot run (it emulates x86); the server has to tolerate that.
9. **Game data** is fetched a file at a time: a few thousand requests on
   entering the world. Fine on a local network; over the internet the round
   trips will add up, and an archive or HTTP/2 will matter. The dev server
   sets `TCP_NODELAY` - without it every request waited 40 ms on a delayed
   ACK; a real web server does not have that problem.

## Roadmap

1. Memory budgets for the browser, and measure what holds the 2.4 GB
   (problem 1).
2. Side-by-side comparison with the native client; fix what the port gets
   wrong (problem 2).
3. Performance. Profile first (Chromium's performance panel). Candidates:
   per-command allocations and `std::variant` dispatch in the replay,
   bind group and state caching, the ~6 MB of small buffers re-uploaded each
   frame, WebAssembly SIMD and LTO, render bundles.
4. Particles as quads; GPU readbacks.
5. Hosting on a server (see below).

## Hosting

The page needs a secure context (HTTPS) for WebGPU and for the shared memory
the threads use, and it must be cross-origin isolated
(`Cross-Origin-Opener-Policy: same-origin`,
`Cross-Origin-Embedder-Policy: require-corp`). So:

- a real web server (Caddy or nginx) with a TLS certificate, those two
  headers, and range requests for `Data/`;
- the relay behind TLS as `wss://`, restricted to the game server's own
  ports - `serve-wasm.py` will connect anywhere it is asked and is for
  localhost only;
- only the build output and the data deployed; there is no need to build on
  the server.

The game data is Blizzard's. Serving it to whoever opens the page is
distributing it. For a private server, put the page behind a login. The
better design is for each player to import their own client's data into
browser storage once; WasmFS has an OPFS backend for exactly that, and
`web_data.cpp` is where it would go.
