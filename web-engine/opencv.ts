/**
 * Loads OpenCV.js from a CDN on first use.
 *
 * The desktop app links against the native OpenCV in its Python venv. The web
 * build pulls the WASM build instead: a 10 MB script (≈3 MB over the wire after
 * brotli) with the WASM binary inlined as base64. jsDelivr marks it immutable
 * for a year, so only the first visit pays the download and compile cost.
 *
 * Two properties of that bundle drive the code below, both measured against the
 * real file rather than assumed:
 *
 *  1. It is wrapped in UMD, and UMD prefers AMD. If `define.amd` exists when the
 *     script evaluates — Monaco ships exactly such a loader — OpenCV registers
 *     itself as an anonymous AMD module that nothing ever requires, `window.cv`
 *     is never assigned, and the load hangs forever. Hiding `define` across the
 *     evaluation forces the plain browser-global branch.
 *
 *  2. `Module.onRuntimeInitialized` is not usable here: the wrapper ends with
 *     `if (typeof Module === 'undefined') var Module = {}`, and because `var` is
 *     hoisted that condition is always true, so the module builds its own config
 *     object and ignores any global one. Readiness is therefore detected by
 *     polling for `cv.Mat`, which appears about a second after evaluation.
 */
const OPENCV_URL = 'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js'

/** Compiling can legitimately take a while on a slow machine, but not forever. */
const READY_TIMEOUT_MS = 120_000
const POLL_INTERVAL_MS = 200

declare global {
  interface Window {
    cv: any
    define: any
  }
}

export interface LoadProgress {
  /** 0–1 when measurable, null for the open-ended compile phase. */
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
let lastProgress: LoadProgress = { progress: 0, message: 'Chargement d’OpenCV.js…' }

const startedAt = performance.now()

function report(update: LoadProgress): void {
  lastProgress = update
  // The page cannot be inspected with devtools while the module compiles, so
  // this trace is the only way to tell a slow load from a stalled one.
  console.info(`[opencv] ${Math.round(performance.now() - startedAt)}ms — ${update.message}`)
  for (const handler of progressHandlers) handler(update)
}

/** Loads the bundle with any AMD loader hidden, restoring it as soon as it has run. */
function injectScript(): Promise<void> {
  return new Promise((resolve, reject) => {
    const previousDefine = window.define
    const hadAmd = typeof previousDefine === 'function' && !!previousDefine.amd
    if (hadAmd) window.define = undefined

    // The AMD check happens during evaluation, so the window stays as narrow as
    // possible — anything else on the page keeps its loader either side of it.
    const restore = () => {
      if (hadAmd) window.define = previousDefine
    }

    const script = document.createElement('script')
    script.src = OPENCV_URL
    script.async = true
    script.onload = () => {
      restore()
      resolve()
    }
    script.onerror = () => {
      restore()
      reject(new Error(`OpenCV.js : téléchargement impossible depuis ${OPENCV_URL}`))
    }
    document.head.appendChild(script)
  })
}

function waitForRuntime(): Promise<any> {
  return new Promise((resolve, reject) => {
    const poll = setInterval(() => {
      if (window.cv?.Mat) {
        clearInterval(poll)
        report({ progress: 1, message: 'OpenCV.js prêt' })
        resolve(window.cv)
        return
      }
      if (performance.now() - startedAt > READY_TIMEOUT_MS) {
        clearInterval(poll)
        // Say which half failed: a missing global means the script never
        // exposed itself, a present one means the WASM module stalled.
        reject(
          new Error(
            window.cv
              ? 'OpenCV.js : le module WASM ne s’est pas initialisé (délai dépassé).'
              : 'OpenCV.js : le script s’est chargé mais n’a rien exposé (window.cv absent).'
          )
        )
      }
    }, POLL_INTERVAL_MS)
  })
}

export function loadOpenCv(onProgress: ProgressHandler = () => {}): Promise<any> {
  progressHandlers.add(onProgress)
  onProgress(lastProgress)

  if (loading) return loading

  loading = (async () => {
    if (window.cv?.Mat) return window.cv
    report({ progress: 0.15, message: 'Téléchargement d’OpenCV.js (~3 Mo)…' })
    await injectScript()
    report({ progress: 0.6, message: 'Compilation du module WASM…' })
    return waitForRuntime()
  })()

  return loading
}

/** Lets a caller stop receiving updates when its component unmounts. */
export function offOpenCvProgress(onProgress: ProgressHandler): void {
  progressHandlers.delete(onProgress)
}
