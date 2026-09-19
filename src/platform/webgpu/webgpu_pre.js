// Asks the browser for a WebGPU device before main() runs.
//
// Getting a device is asynchronous and the client's startup is not, so the
// page gets one first and hands it over as Module.preinitializedWebGPUDevice,
// where emscripten_webgpu_get_device() finds it (see device.cpp).
//
// The optional features the translation layer uses are requested when the
// adapter has them, along with the adapter's full limits rather than WebGPU's
// defaults: the renderer's storage buffers and bind groups need them.

// At most eight cores, as far as the client can tell. Its thread pools size
// themselves from the core count, and every thread here is a Web Worker - a
// wide machine would start around forty. Top level rather than in preRun, so
// the worker threads, which run this file too, see the same number.
if (typeof navigator !== 'undefined' && navigator.hardwareConcurrency > 8) {
  Object.defineProperty(navigator, 'hardwareConcurrency', { value: 8, configurable: true });
}

Module['preRun'] = Module['preRun'] || [];
Module['preRun'].push(function () {
  // ?capture=N logs frame N as a PNG (see capture.cpp).
  var capture = new URLSearchParams(location.search).get('capture');
  if (capture !== null) ENV['WOWEE_CAPTURE_FRAME'] = capture;
  if (new URLSearchParams(location.search).has('offscreen')) ENV['WOWEE_OFFSCREEN'] = '1';
  if (new URLSearchParams(location.search).has('eager')) ENV['WOWEE_EAGER_PIPELINES'] = '1';
  // Debug: ?autologin=user:pass@host[:port] and ?enterworld drive the login
  // and character screens (auth_screen.cpp, character_screen.cpp).
  var auto = new URLSearchParams(location.search).get('autologin');
  if (auto !== null) ENV['WOWEE_AUTOLOGIN'] = auto;
  if (new URLSearchParams(location.search).has('enterworld')) ENV['WOWEE_AUTO_ENTER'] = '1';
  // ?env=NAME=VALUE,NAME=VALUE: settings the client reads from the
  // environment, such as the cache budgets or WOWEE_FRAME_PROFILE.
  var env = new URLSearchParams(location.search).get('env');
  if (env !== null) env.split(',').forEach(function (pair) {
    var eq = pair.indexOf('=');
    if (eq > 0) ENV[pair.slice(0, eq)] = pair.slice(eq + 1);
  });
  // ?log=debug|info|warn|error: the log level (logger.cpp).
  var log = new URLSearchParams(location.search).get('log');
  if (log !== null) ENV['WOWEE_LOG_LEVEL'] = log;

  addRunDependency('webgpu-device');
  (async function () {
    if (!navigator.gpu) {
      throw new Error('This browser has no WebGPU (navigator.gpu is missing).');
    }
    var adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('WebGPU: no adapter.');
    var ai = adapter.info || {};
    console.log('WebGPU adapter: ' + [ai.vendor, ai.architecture, ai.device, ai.description].filter(Boolean).join(' / '));

    var wanted = ['texture-compression-bc', 'float32-filterable', 'depth32float-stencil8'];
    var features = wanted.filter(function (f) { return adapter.features.has(f); });

    // The adapter's own limits first. Not every browser will make a device
    // at its absolute maximums - Firefox answers "OperationError: Not enough
    // memory left" for the largest buffer sizes - so then limits capped to what
    // the renderer can use, and last WebGPU's defaults.
    var full = {}, capped = {};
    var caps = { maxBufferSize: 1 << 30, maxStorageBufferBindingSize: 1 << 30,
                 maxUniformBufferBindingSize: 1 << 16 };
    for (var key in adapter.limits) {
      var v = adapter.limits[key];
      if (typeof v !== 'number') continue;
      full[key] = v;
      capped[key] = key in caps ? Math.min(v, caps[key]) : v;
    }
    var device = null, lastError = null;
    for (var limits of [full, capped, undefined]) {
      try {
        device = await adapter.requestDevice({ requiredFeatures: features, requiredLimits: limits });
        break;
      } catch (e) {
        lastError = e;
        console.warn('WebGPU: device with ' + (limits === full ? 'full' : limits ? 'capped' : 'default') +
                     ' limits refused: ' + e);
      }
    }
    if (!device) throw lastError;
    device.lost.then(function (info) {
      console.error('WebGPU device lost: ' + info.reason + ' - ' + info.message);
    });
    // Validation errors are reported once per distinct message, so a bad
    // pipeline used every frame does not bury everything else.
    var seen = new Set();
    device.addEventListener('uncapturederror', function (e) {
      var msg = e.error.message;
      if (seen.has(msg)) return;
      seen.add(msg);
      console.error('WebGPU error: ' + msg);
    });

    Module['preinitializedWebGPUDevice'] = device;
    removeRunDependency('webgpu-device');
  })().catch(function (e) {
    console.error(e);
    if (Module['setStatus']) Module['setStatus'](String(e));
  });
});
