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

Not working or not right yet - see *Known problems*:

- No loading screen, then a ~30 s hang on entering the world.
- Some textures and lighting look off.
- Particles are single pixels.
- Firefox: the device request fails ("Not enough memory left"). Chromium only.

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
| Game data | `src/platform/web_data.cpp` | `/Data` is a WasmFS fetch mount: each file is downloaded from the server the first time it is read. |
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

A full headless run into the world and a frame of it:

```bash
chromium --headless=new --enable-unsafe-webgpu --enable-logging=stderr \
  'http://127.0.0.1:8080/wowee.html?offscreen&autologin=USER:PASS@HOST&enterworld&capture=1200' \
  2>&1 | tee run.log | grep -o 'WOWEE_CAPTURE [A-Za-z0-9+/=]*' | cut -d' ' -f2 | base64 -d > frame.png
```

Headless Chromium draws on the CPU (SwiftShader), so it is far slower than a
real GPU; use it for correctness, not for frame rates.

The layer logs a pipeline or shader WebGPU refuses once, with the reason
(`WebGPU pipeline ... refused`, `WebGPU shader ... does not compile`), and
skips draws with a refused pipeline rather than losing the frame.

## Known problems

1. **No loading screen; a long hang entering the world.** The browser shows a
   frame only when the main loop returns to it, and the world load is one
   blocking loop that draws its own loading frames. Fix: split the load into
   steps run from successive frames.
2. **Textures and lighting partly wrong.** Not yet separated into port bugs
   and client bugs. Compare the same spot in the native client and the
   browser. Suspects on the port side: depth textures read through a nearest
   sampler (`VkSampler_T::ensureNearest`), the `textureQueryLod` stand-in in
   `wasm_shaders.py`, canvas sRGB handling.
3. **Particles are one pixel.** WebGPU has no point size. The shader tool
   already routes `gl_PointSize`/`gl_PointCoord` through varyings (locations
   14/15); what is left is expanding each point to a quad (wrap `main`,
   instanced draw of 6 vertices per point).
4. **No GPU-to-CPU readback.** `vkCmdCopyImageToBuffer` and copies into
   host-visible buffers log a warning and do nothing. Needs `mapAsync` and
   copying into the CPU copy when it completes.
5. **Some compute passes cannot run**: FSR2 and HiZ use read-write storage
   texture formats WebGPU does not allow (rgba16float, rg16float).
6. **Firefox** refuses the device. The fallback chain in `webgpu_pre.js` did
   not help; not investigated further.
7. **Warden** cannot run (it emulates x86); the server has to tolerate that.
8. **Game data** is fetched a file at a time: a few thousand requests on
   entering the world. The fetch backend also keeps every file it has
   downloaded in memory for the session.

## Roadmap

1. Loading screen: make the world load incremental (fixes problem 1).
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
