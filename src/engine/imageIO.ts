const imageElCache = new Map<string, Promise<HTMLImageElement>>()

function loadImageElement(src: string): Promise<HTMLImageElement> {
  let p = imageElCache.get(src)
  if (p) return p
  p = new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`Impossible de charger l'image: ${src}`))
    img.src = src
  })
  imageElCache.set(src, p)
  return p
}

/** Loads an image URL (sample path or data: URL) into a fresh BGR cv.Mat. Caller owns the returned Mat. */
export async function loadImageAsMat(cv: any, src: string): Promise<any> {
  const img = await loadImageElement(src)
  const canvas = document.createElement('canvas')
  canvas.width = img.naturalWidth
  canvas.height = img.naturalHeight
  const ctx2d = canvas.getContext('2d')!
  ctx2d.drawImage(img, 0, 0)
  const rgba = cv.imread(canvas)
  const bgr = new cv.Mat()
  cv.cvtColor(rgba, bgr, cv.COLOR_RGBA2BGR)
  rgba.delete()
  return bgr
}

/** Renders a cv.Mat (any channel count) to a data URL for <img> / <canvas> display. */
export function matToDataUrl(cv: any, mat: any): string {
  const canvas = document.createElement('canvas')
  canvas.width = mat.cols
  canvas.height = mat.rows
  let toShow = mat
  let needsDelete = false
  if (mat.channels() === 1) {
    toShow = new cv.Mat()
    cv.cvtColor(mat, toShow, cv.COLOR_GRAY2RGBA)
    needsDelete = true
  } else if (mat.channels() === 3) {
    toShow = new cv.Mat()
    cv.cvtColor(mat, toShow, cv.COLOR_BGR2RGBA)
    needsDelete = true
  }
  cv.imshow(canvas, toShow)
  if (needsDelete) toShow.delete()
  return canvas.toDataURL('image/png')
}
