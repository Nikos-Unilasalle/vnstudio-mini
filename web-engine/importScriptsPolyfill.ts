/**
 * Restores `importScripts` for third-party libraries that call it directly.
 *
 * worker.ts has to be a *module* worker (it uses `import`), and module
 * workers make the native `importScripts` throw — the function reference
 * still exists (so `typeof importScripts === 'function'` checks pass and
 * libraries take their "we're in a worker" branch), calling it is what's
 * disallowed. `opencv.ts` sidesteps this for its own load with a hand-patched
 * `import()` (see its doc comment), but that trick doesn't scale to
 * third-party bundles we don't control the source of — MediaPipe's
 * `@mediapipe/tasks-vision` calls `importScripts()` unconditionally in any
 * worker context to load its own WASM runtime.
 *
 * This replaces `importScripts` with a synchronous-XHR + indirect-`eval`
 * polyfill: synchronous XHR is deprecated on the main thread but still fully
 * supported in workers (it's how the native implementation works internally),
 * and indirect `eval` — `(0, eval)(code)` rather than `eval(code)` — always
 * runs as non-strict, global-scope code regardless of the module's own strict
 * mode, so a UMD wrapper relying on top-level `this` being the global object
 * (the same issue opencv.ts's glue has) works correctly here too, with no
 * per-library patching needed.
 */
;(self as any).importScripts = function importScriptsPolyfill(...urls: string[]) {
  for (const url of urls) {
    const xhr = new XMLHttpRequest()
    xhr.open('GET', url, false)
    xhr.send(null)
    if (xhr.status !== 200 && xhr.status !== 0) {
      throw new Error(`importScripts polyfill: HTTP ${xhr.status} fetching ${url}`)
    }
    // eslint-disable-next-line no-eval
    ;(0, eval)(xhr.responseText)
  }
}
