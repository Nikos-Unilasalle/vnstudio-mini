/**
 * Loads OpenCV.js from a CDN on first use.
 *
 * The desktop app links against the native OpenCV in its Python venv. The web
 * build pulls the WASM build instead — ~9 MB, cached by the browser after the
 * first visit, and fetched lazily so the UI paints before it lands.
 */
// jsDelivr rather than docs.opencv.org: the latter only keeps some versions
// online and serves no CORS headers, so a pinned build there can 404 later.
const OPENCV_URL = 'https://cdn.jsdelivr.net/npm/@techstark/opencv-js@4.10.0-release.1/dist/opencv.js'

declare global {
  interface Window {
    cv: any
  }
}

let loading: Promise<any> | null = null

export function loadOpenCv(): Promise<any> {
  if (loading) return loading

  loading = new Promise((resolve, reject) => {
    if (window.cv?.Mat) {
      resolve(window.cv)
      return
    }

    const script = document.createElement('script')
    script.src = OPENCV_URL
    script.async = true
    script.onerror = () => reject(new Error('Échec du chargement d’OpenCV.js'))
    script.onload = () => {
      const cv = window.cv
      if (!cv) {
        reject(new Error('opencv.js chargé mais window.cv absent'))
        return
      }
      // Three shapes in the wild: already initialised, a promise for the module,
      // or a module that signals readiness through onRuntimeInitialized.
      if (cv.Mat) resolve(cv)
      else if (typeof cv.then === 'function') cv.then((ready: any) => {
        window.cv = ready
        resolve(ready)
      })
      else cv.onRuntimeInitialized = () => resolve(window.cv)
    }
    document.head.appendChild(script)
  })

  return loading
}
