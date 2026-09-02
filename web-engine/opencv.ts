/**
 * Loads OpenCV.js from a CDN on first use.
 *
 * The desktop app links against the native OpenCV in its Python venv. The web
 * build pulls the WASM build instead: a ~10 MB script with the WASM binary
 * inlined as base64, so the first visit pays a download and a compile.
 *
 * Three properties of that bundle drive the code below, each verified against
 * the real file rather than assumed:
 *
 *  1. Readiness must be signalled through a `Module` object created *before*
 *     the script runs. This is the documented Emscripten pattern: the bundle
 *     ends with `if (typeof Module === 'undefined') Module = {}` and hands that
 *     object to the factory, so a global defined up front is adopted and its
 *     `onRuntimeInitialized` fires. Attaching the callback afterwards is a race
 *     that silently never fires.
 *
 *     This is why the build matters. The @techstark mirror ships the same line
 *     as `var Module = {}` — and because `var` is hoisted, that condition is
 *     always true, so it builds its own config object and ignores any global.
 *     With that mirror the callback can never be delivered. The official build
 *     is used precisely because it honours a pre-defined `Module`.
 *
 *  2. The bundle is wrapped in UMD, and UMD prefers AMD. If `define.amd` exists
 *     while the script evaluates — Monaco ships exactly such a loader — OpenCV
 *     registers as an anonymous AMD module that nothing requires, `window.cv` is
 *     never assigned, and the load hangs. `define` is hidden across evaluation.
 *
 *  3. `script.onload` fires well before the module is usable, because the WASM
 *     is compiled after evaluation.
 */
const OPENCV_URL = 'https://docs.opencv.org/4.9.0/opencv.js'

/** Compiling can legitimately take a while on a slow machine, but not forever. */
const READY_TIMEOUT_MS = 120_000
const POLL_INTERVAL_MS = 250

declare global {
  interface Window {
    cv: any
    Module: any
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

export function loadOpenCv(onProgress: ProgressHandler = () => {}): Promise<any> {
  progressHandlers.add(onProgress)
  onProgress(lastProgress)

  if (loading) return loading

  loading = new Promise<any>((resolve, reject) => {
    if (window.cv?.Mat) {
      resolve(window.cv)
      return
    }

    let settled = false
    const succeed = (cv: any) => {
      if (settled) return
      settled = true
      clearInterval(poll)
      window.cv = cv
      report({ progress: 1, message: 'OpenCV.js prêt' })
      resolve(cv)
    }
    const failWith = (message: string) => {
      if (settled) return
      settled = true
      clearInterval(poll)
      reject(new Error(message))
    }

    // Set up the config object first — the bundle adopts this exact object.
    const runtime: any = {
      onRuntimeInitialized: () => succeed(runtime),
      onAbort: (reason: unknown) => failWith(`OpenCV.js : le module WASM a abandonné (${String(reason)})`),
      // Emscripten writes initialisation failures here rather than throwing.
      printErr: (text: string) => console.error(`[opencv] ${text}`),
    }
    window.Module = runtime

    // Belt and braces: if the callback is somehow missed, polling still detects
    // the module, and the timeout reports which half failed.
    const poll = setInterval(() => {
      if (window.cv?.Mat) {
        succeed(window.cv)
        return
      }
      if (runtime.Mat) {
        succeed(runtime)
        return
      }
      if (performance.now() - startedAt > READY_TIMEOUT_MS) {
        failWith(
          window.cv
            ? 'OpenCV.js : le module WASM ne s’est pas initialisé (délai dépassé).'
            : 'OpenCV.js : le script s’est chargé mais n’a rien exposé (window.cv absent).'
        )
      }
    }, POLL_INTERVAL_MS)

    report({ progress: 0.15, message: 'Téléchargement d’OpenCV.js (~9 Mo)…' })
    injectScript()
      .then(() => report({ progress: 0.6, message: 'Compilation du module WASM…' }))
      .catch((error: Error) => failWith(error.message))
  })

  return loading
}

/** Lets a caller stop receiving updates when its component unmounts. */
export function offOpenCvProgress(onProgress: ProgressHandler): void {
  progressHandlers.delete(onProgress)
}
