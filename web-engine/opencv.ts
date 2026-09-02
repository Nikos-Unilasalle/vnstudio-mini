/**
 * Loads OpenCV (WASM) on first use.
 *
 * The desktop app links against the native OpenCV in its Python venv. The web
 * build uses a distribution that ships the WASM binary as a **separate file**
 * rather than inlining it as base64:
 *
 *   opencv.js    0.4 MB   emscripten glue
 *   opencv.wasm  7.0 MB   the module itself
 *
 * That split is the whole point. The common single-file builds bundle both into
 * one ~10 MB script, so the browser must parse 10 MB of JavaScript and decode a
 * base64 payload on the main thread before anything happens, and readiness can
 * only be observed through `Module.onRuntimeInitialized` — a callback that is
 * easy to miss and impossible to attach on some mirrors, which leaves the app
 * hanging with no error. Here the binary is fetched separately (so download
 * progress is real) and handed to the glue as `Module.wasmBinary`, after which
 * the module initialises *synchronously*: `cv.Mat` exists the moment the script
 * finishes evaluating. There is no callback to miss and no race to lose.
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

declare global {
  interface Window {
    cv: any
    define: any
    __opencvWasmBinary?: Uint8Array
  }
}

export interface LoadProgress {
  /** 0–1 when measurable, null when the phase has no meaningful ratio. */
  progress: number | null
  message: string
}

type ProgressHandler = (progress: LoadProgress) => void

let loading: Promise<any> | null = null

/**
 * Every caller gets progress, not just the one that started the load. React
 * StrictMode mounts the engine hook twice in development and tears the first
 * one down immediately; reporting only to it would freeze the bar.
 */
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
  const NEEDLE = 'let Module = {};'
  if (!source.includes(NEEDLE)) {
    throw new Error('OpenCV : format du script inattendu (le point d’injection du binaire WASM a changé).')
  }
  return source.replace(NEEDLE, 'let Module = { wasmBinary: window.__opencvWasmBinary };')
}

/**
 * The glue is wrapped in UMD, and UMD prefers AMD. Monaco ships an AMD loader,
 * and with `define.amd` present OpenCV would register as an anonymous module
 * that nothing requires — leaving `window.cv` unassigned forever. Hiding
 * `define` across the evaluation forces the plain browser-global branch.
 */
function evaluateGlue(source: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const previousDefine = window.define
    const hadAmd = typeof previousDefine === 'function' && !!previousDefine.amd
    if (hadAmd) window.define = undefined
    const restore = () => {
      if (hadAmd) window.define = previousDefine
    }

    const url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }))
    const script = document.createElement('script')
    script.src = url
    script.onload = () => {
      restore()
      URL.revokeObjectURL(url)
      resolve()
    }
    script.onerror = () => {
      restore()
      URL.revokeObjectURL(url)
      reject(new Error('OpenCV : le script n’a pas pu être exécuté.'))
    }
    document.head.appendChild(script)
  })
}

export function loadOpenCv(onProgress: ProgressHandler = () => {}): Promise<any> {
  progressHandlers.add(onProgress)
  onProgress(lastProgress)

  if (loading) return loading

  loading = (async () => {
    if (window.cv?.Mat) return window.cv

    report({ progress: 0, message: 'Téléchargement d’OpenCV…' })
    const [binary, glue] = await Promise.all([fetchWasmBinary(), fetchPatchedGlue()])
    window.__opencvWasmBinary = binary

    report({ progress: 0.9, message: 'Initialisation du module WASM…' })
    await evaluateGlue(glue)

    // With the binary preloaded the module is ready as soon as the script has
    // run, so a missing Mat here is a real failure rather than a slow start.
    if (!window.cv?.Mat) {
      throw new Error('OpenCV : le module s’est chargé mais reste inutilisable (cv.Mat absent).')
    }

    // The binary is copied into the WASM heap during init; drop our reference so
    // 7 MB is not pinned for the lifetime of the page.
    delete window.__opencvWasmBinary

    report({ progress: 1, message: 'OpenCV prêt' })
    return window.cv
  })()

  return loading
}

/** Lets a caller stop receiving updates when its component unmounts. */
export function offOpenCvProgress(onProgress: ProgressHandler): void {
  progressHandlers.delete(onProgress)
}
