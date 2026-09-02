/**
 * Canvas helpers for the worker.
 *
 * Everything in `web-engine/` now runs inside the Web Worker spawned by
 * `shims/useVisionEngine.ts` (see worker.ts) — never on the main thread — so
 * these use `OffscreenCanvas` unconditionally rather than branching on
 * context. `OffscreenCanvas` has no `toDataURL`; `convertToBlob` is the async
 * replacement, hence every caller of these two helpers is (and must stay) async.
 */

export function makeCanvas(width: number, height: number): OffscreenCanvas {
  return new OffscreenCanvas(width, height)
}

/**
 * Stand-ins for `cv.imread`/`cv.imshow`. This OpenCV.js build's own JS glue
 * for both starts with `instanceof HTMLImageElement` / `instanceof
 * HTMLCanvasElement` checks — plain `ReferenceError`s in a worker, since
 * neither global exists there, regardless of what's actually passed in. Both
 * replacements below run the exact same pixel math the glue does after its
 * (here-skipped) DOM validation, just against `OffscreenCanvas`.
 */
export function matFromCanvas(cv: any, canvas: OffscreenCanvas): any {
  const ctx = canvas.getContext('2d')!
  const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height)
  return cv.matFromImageData(imgData)
}

export function drawMatToCanvas(cv: any, canvas: OffscreenCanvas, mat: any): void {
  const img = new cv.Mat()
  const depth = mat.type() % 8
  const scale = depth <= cv.CV_8S ? 1 : depth <= cv.CV_32S ? 1 / 256 : 255
  const shift = depth === cv.CV_8S || depth === cv.CV_16S ? 128 : 0
  mat.convertTo(img, cv.CV_8U, scale, shift)
  switch (img.type()) {
    case cv.CV_8UC1:
      cv.cvtColor(img, img, cv.COLOR_GRAY2RGBA)
      break
    case cv.CV_8UC3:
      cv.cvtColor(img, img, cv.COLOR_RGB2RGBA)
      break
    case cv.CV_8UC4:
      break
    default:
      img.delete()
      throw new Error('Bad number of channels (Source image must have 1, 3 or 4 channels)')
  }
  const imgData = new ImageData(new Uint8ClampedArray(img.data), img.cols, img.rows)
  canvas.width = imgData.width
  canvas.height = imgData.height
  canvas.getContext('2d')!.putImageData(imgData, 0, 0)
  img.delete()
}

function bufferToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf)
  let binary = ''
  const chunkSize = 0x8000
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize))
  }
  return btoa(binary)
}

/** Base64 payload only, no `data:` prefix — matches what `<img>` callers reassemble. */
export async function canvasToBase64(canvas: OffscreenCanvas, quality = 0.8): Promise<string> {
  const blob = await canvas.convertToBlob({ type: 'image/jpeg', quality })
  return bufferToBase64(await blob.arrayBuffer())
}

/** Full `data:image/jpeg;base64,...` URL, for callers that hand the string straight to an `<img src>`. */
export async function canvasToDataUrl(canvas: OffscreenCanvas, quality = 0.8): Promise<string> {
  return `data:image/jpeg;base64,${await canvasToBase64(canvas, quality)}`
}
