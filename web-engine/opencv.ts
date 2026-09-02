/**
 * Loads OpenCV.js from a CDN on first use.
 *
 * The desktop app links against the native OpenCV in its Python venv. The web
 * build pulls the WASM build instead. It is a 10 MB script (≈3 MB over the wire
 * after brotli) with the WASM binary inlined as base64, so the browser has to
 * parse the script, decode the payload and compile the module — several seconds
 * of busy CPU on first visit. The CDN marks it immutable for a year, so every
 * later visit is served from disk cache and initialises far faster.
 *
 * jsDelivr rather than docs.opencv.org: the latter serves no CORS headers, and
 * without those the download cannot be streamed for progress reporting.
 */
const OPENCV_URL = 'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js'

/** Uncompressed size, used to show progress when the server omits Content-Length. */
const APPROX_BYTES = 10_378_215

/** Compiling the module can legitimately take a while on a slow machine, but not forever. */
const READY_TIMEOUT_MS = 180_000

declare global {
  interface Window {
    cv: any
  }
}

export interface LoadProgress {
  /** 0–1 across the whole load, or null once the phase is not measurable. */
  progress: number | null
  message: string
}

type ProgressHandler = (progress: LoadProgress) => void

let loading: Promise<any> | null = null

/** Downloading is most of the wall-clock time but not all of it; leave room for compiling. */
const DOWNLOAD_SHARE = 0.7

async function fetchScript(onProgress: ProgressHandler): Promise<string> {
  const response = await fetch(OPENCV_URL)
  if (!response.ok) throw new Error(`OpenCV.js : HTTP ${response.status}`)

  // Content-Length reflects the compressed size while the body reads back
  // decompressed, so treat the ratio as approximate and clamp it.
  const declared = Number(response.headers.get('content-length')) || 0
  const total = declared > APPROX_BYTES / 2 ? declared : APPROX_BYTES

  if (!response.body) return response.text()

  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let received = 0
  // The body arrives in hundreds of small chunks; reporting each one would
  // re-render the notification bar far more often than a human can read it.
  let lastReported = 0

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.length
    const ratio = Math.min(1, received / total)
    if (ratio - lastReported < 0.05) continue
    lastReported = ratio
    onProgress({
      progress: ratio * DOWNLOAD_SHARE,
      message: `Téléchargement d’OpenCV.js… ${Math.round(ratio * 100)} %`,
    })
  }

  const merged = new Uint8Array(received)
  let offset = 0
  for (const chunk of chunks) {
    merged.set(chunk, offset)
    offset += chunk.length
  }
  return new TextDecoder().decode(merged)
}

function runScript(source: string): void {
  // A blob URL keeps this out of the HTML and lets the browser treat it as a
  // normal classic script, which is what the UMD wrapper expects.
  const url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }))
  const script = document.createElement('script')
  script.src = url
  script.onload = () => URL.revokeObjectURL(url)
  document.head.appendChild(script)
}

/**
 * Emscripten modules signal readiness through `onRuntimeInitialized`, but that
 * callback is lost if the runtime finished before we could attach it — which
 * happens with an inlined WASM payload. Polling covers both orderings.
 */
function waitForRuntime(onProgress: ProgressHandler): Promise<any> {
  return new Promise((resolve, reject) => {
    const started = Date.now()
    let announcedCompiling = false

    const settle = () => {
      clearInterval(poll)
      onProgress({ progress: 1, message: 'OpenCV.js prêt' })
      resolve(window.cv)
    }

    const poll = setInterval(() => {
      if (window.cv?.Mat) {
        settle()
        return
      }
      if (Date.now() - started > READY_TIMEOUT_MS) {
        clearInterval(poll)
        reject(new Error('OpenCV.js : délai dépassé pendant l’initialisation du module WASM.'))
        return
      }
      if (!announcedCompiling) {
        announcedCompiling = true
        onProgress({ progress: null, message: 'Compilation du module WASM…' })
      }
    }, 250)

    // Belt and braces: if the callback does fire, resolve without waiting for
    // the next poll tick.
    const attach = setInterval(() => {
      if (!window.cv || window.cv.Mat) {
        clearInterval(attach)
        return
      }
      if (typeof window.cv.then === 'function') {
        clearInterval(attach)
        window.cv.then((ready: any) => {
          window.cv = ready
          settle()
        })
      } else if (!window.cv.onRuntimeInitialized) {
        window.cv.onRuntimeInitialized = settle
        clearInterval(attach)
      }
    }, 50)
  })
}

export function loadOpenCv(onProgress: ProgressHandler = () => {}): Promise<any> {
  if (loading) return loading

  loading = (async () => {
    if (window.cv?.Mat) return window.cv

    onProgress({ progress: 0, message: 'Téléchargement d’OpenCV.js…' })
    const source = await fetchScript(onProgress)

    onProgress({ progress: DOWNLOAD_SHARE, message: 'Compilation du module WASM…' })
    runScript(source)

    return waitForRuntime(onProgress)
  })()

  return loading
}
