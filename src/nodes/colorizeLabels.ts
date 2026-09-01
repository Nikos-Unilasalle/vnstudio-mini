import { labelColor } from './labelStats'

/** Renders a CV_32S label matrix as a BGR8U3 colorized preview (fresh Mat, caller owns it). */
export function colorizeLabels(cv: any, labelsMat: any): any {
  const w = labelsMat.cols
  const h = labelsMat.rows
  const src = labelsMat.data32S as Int32Array
  const out = new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(0, 0, 0))
  const dst = out.data as Uint8Array
  const colorCache = new Map<number, [number, number, number]>()

  for (let i = 0; i < w * h; i++) {
    const label = src[i]
    if (label <= 0) continue
    let rgb = colorCache.get(label)
    if (!rgb) {
      rgb = labelColor(label)
      colorCache.set(label, rgb)
    }
    const off = i * 3
    dst[off] = rgb[2] // B
    dst[off + 1] = rgb[1] // G
    dst[off + 2] = rgb[0] // R
  }
  return out
}
