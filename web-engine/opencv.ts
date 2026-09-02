/**
 * Loads OpenCV (WASM) on first use — worker-side only (see worker.ts).
 *
 * The web build uses a distribution that ships the WASM binary as a
 * **separate file** rather than inlining it as base64:
 *
 *   opencv.js    0.4 MB   emscripten glue
 *   opencv.wasm  7.0 MB   the module itself
 *
 * That split gives real download progress instead of one opaque ~10 MB
 * script. But the glue's own instantiation is still Emscripten's classic
 * *synchronous* path (`new WebAssembly.Module(binary)` +
 * `new WebAssembly.Instance(...)`, not `WebAssembly.instantiate`) — this
 * build's export wiring assumes the result is available the instant the glue
 * finishes evaluating, so an async `instantiateWasm` override breaks it
 * (`globalCtors is not a function`, thrown before the async instantiate has
 * even resolved). That's fine *in a worker*: blocking only stalls this
 * thread, never the page, so we let it block.
 *
 * Loading it is more contrary than it should be:
 * - `importScripts` (the obvious synchronous loader) is disallowed in module
 *   workers, and Vite always serves worker.ts as a module worker in dev
 *   (`?worker` only gets you a true classic bundle in the production build) —
 *   so relying on it breaks `npm run dev` even though it works once deployed.
 * - Dynamic `import()` works in both, but ES modules are strict-mode with
 *   `this === undefined` at the top level, and the glue's UMD wrapper does
 *   `(function(root, factory){ ... })(this, function(){ ... })`, needing
 *   `root` to be the global object to attach `cv` to. So the glue text is
 *   patched to invoke itself with `self` instead of relying on `this`.
 * - That same wrapper also special-cases `typeof importScripts === 'function'`
 *   — true in *any* worker, module or not, since the global exists even where
 *   calling it is forbidden — as "construct the module explicitly": it
 *   assigns the *factory function* to `cv` rather than invoking it, unlike
 *   every other branch. So `cv` has to be invoked once more by hand below.
 * - The resulting `cv` object has its own `.then()` method (an Emscripten
 *   "run this once ready" convenience API), which makes it a thenable. Ever
 *   `return`ed or `resolve()`d bare, the JS promise machinery chains onto
 *   *that* `.then()` instead of settling with the object — and since our own
 *   init already finished by the time we'd call it, whatever queue it drains
 *   from is empty, so the callback never fires and the promise hangs forever
 *   with no error. `loadOpenCv` resolves to `{ cv }` to sidestep this.
 *
 * This distribution is OpenCV 4.3, which omits `HuMoments` and `polylines` from
 * its JS bindings — see huMoments() and drawPolyline() in cvUtils.ts for the
 * replacements.
 */
const PACKAGE_BASE = 'https://cdn.jsdelivr.net/npm/opencv-wasm@4.3.0-10'
const WASM_URL = `${PACKAGE_BASE}/opencv.wasm`
const GLUE_URL = `${PACKAGE_BASE}/opencv.js`

/** Approximate size of the binary, used when the CDN omits Content-Length. */
const APPROX_WASM_BYTES = 6_961_000

// `lib: webworker` can't coexist with the app's `lib: dom` in one tsconfig, so
// this is typed loosely rather than as a real `DedicatedWorkerGlobalScope`.
declare const self: {
  cv: any
  define: any
  __opencvWasmBinary?: Uint8Array
}

export interface LoadProgress {
  /** 0–1 when measurable, null when the phase has no meaningful ratio. */
  progress: number | null
  message: string
}

type ProgressHandler = (progress: LoadProgress) => void

// Boxed as `{ cv }` rather than `Promise<cv>` — see loadOpenCv's doc comment.
let loading: Promise<{ cv: any }> | null = null

const progressHandlers = new Set<ProgressHandler>()
let lastProgress: LoadProgress = { progress: 0, message: 'Chargement d’OpenCV…' }

const startedAt = performance.now()

function report(update: LoadProgress): void {
  lastProgress = update
  console.info(`[opencv] ${Math.round(performance.now() - startedAt)}ms — ${update.message}`)
  for (const handler of progressHandlers) handler(update)
}

/** Downloading the binary is the bulk of the wait; leave room for the glue and init. */
const DOWNLOAD_SHARE = 0.85

async function fetchWasmBinary(): Promise<Uint8Array> {
  const response = await fetch(WASM_URL)
  if (!response.ok) throw new Error(`OpenCV : téléchargement du module WASM impossible (HTTP ${response.status})`)

  const declared = Number(response.headers.get('content-length')) || 0
  const total = declared > 0 ? declared : APPROX_WASM_BYTES

  if (!response.body) return new Uint8Array(await response.arrayBuffer())

  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let received = 0
  // Hundreds of small chunks arrive; reporting each would re-render the
  // notification far more often than anyone can read it.
  let lastReported = 0

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.length
    const ratio = Math.min(1, received / total)
    if (ratio - lastReported < 0.05) continue
    lastReported = ratio
    report({ progress: ratio * DOWNLOAD_SHARE, message: `Téléchargement d’OpenCV… ${Math.round(ratio * 100)} %` })
  }

  const binary = new Uint8Array(received)
  let offset = 0
  for (const chunk of chunks) {
    binary.set(chunk, offset)
    offset += chunk.length
  }
  return binary
}

/**
 * The glue hardcodes `let Module = {}` at file scope, so there is no way to hand
 * it a config object from outside — the binary has to be patched in. Without it
 * the module tries a synchronous XHR for the .wasm, which browsers refuse, and
 * aborts with "sync fetching of the wasm failed".
 */
async function fetchPatchedGlue(): Promise<string> {
  const response = await fetch(GLUE_URL)
  if (!response.ok) throw new Error(`OpenCV : téléchargement du script impossible (HTTP ${response.status})`)

  const source = await response.text()
  const MODULE_NEEDLE = 'let Module = {};'
  if (!source.includes(MODULE_NEEDLE)) {
    throw new Error('OpenCV : format du script inattendu (le point d’injection du binaire WASM a changé).')
  }
  // `this` is undefined at the top level of an ES module — see the file doc
  // comment for why the UMD wrapper needs `self` instead.
  const UMD_NEEDLE = '}(this, function() {'
  if (!source.includes(UMD_NEEDLE)) {
    throw new Error('OpenCV : format du script inattendu (l’invocation du wrapper UMD a changé).')
  }
  return source
    .replace(MODULE_NEEDLE, 'let Module = { wasmBinary: self.__opencvWasmBinary };')
    .replace(UMD_NEEDLE, '}(self, function() {')
}

/**
 * Resolves to `{ cv }`, not `cv` directly — the module has its own `.then()`
 * method (see the comment below), so returning or resolving with it bare
 * would make every promise in the chain treat it as a thenable and hang
 * forever instead of settling. Callers do `const { cv } = await loadOpenCv()`.
 */
export function loadOpenCv(onProgress: ProgressHandler = () => {}): Promise<{ cv: any }> {
  progressHandlers.add(onProgress)
  onProgress(lastProgress)

  if (loading) return loading

  // `cv` (the Emscripten Module) exposes its own `.then()` — Emscripten's
  // "call this once the runtime is ready" convenience API. Returning it
  // directly from an async function makes the JS promise machinery treat it
  // as a thenable and chain onto *that* `.then()` instead of resolving with
  // the object, which never settles our promise (that API expects to be
  // subscribed before init finishes, not polled after — by the time we'd
  // call it, whatever queue it drains from is already empty). Boxing it
  // sidesteps the auto-unwrapping entirely.
  loading = (async () => {
    if (self.cv?.Mat) return { cv: self.cv }

    report({ progress: 0, message: 'Téléchargement d’OpenCV…' })
    const [binary, glue] = await Promise.all([fetchWasmBinary(), fetchPatchedGlue()])
    self.__opencvWasmBinary = binary

    report({ progress: 0.9, message: 'Initialisation du module WASM…' })
    // Blocks this worker thread only — see the file doc comment above.
    const url = URL.createObjectURL(new Blob([glue], { type: 'text/javascript' }))
    try {
      await import(/* @vite-ignore */ url)
    } finally {
      URL.revokeObjectURL(url)
    }

    // See the file doc comment: the glue's worker-mode branch hands us the
    // factory function itself rather than its result.
    if (typeof self.cv === 'function') self.cv = self.cv()

    if (!self.cv?.Mat) {
      throw new Error('OpenCV : le module s’est chargé mais reste inutilisable (cv.Mat absent).')
    }

    // The binary is copied into the WASM heap during init; drop our reference so
    // 7 MB is not pinned for the lifetime of the worker.
    delete self.__opencvWasmBinary

    report({ progress: 1, message: 'OpenCV prêt' })
    return { cv: self.cv }
  })()

  return loading
}

/** Lets a caller stop receiving updates when its component unmounts. */
export function offOpenCvProgress(onProgress: ProgressHandler): void {
  progressHandlers.delete(onProgress)
}
