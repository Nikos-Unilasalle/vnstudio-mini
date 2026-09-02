/** Shared OpenCV.js helpers used across the browser node implementations. */
import { makeCanvas, canvasToBase64, drawMatToCanvas } from './canvasCompat'

/** True when the value is a live cv.Mat rather than a scalar/dict/list payload. */
export function isMat(v: unknown): boolean {
  return !!v && typeof v === 'object' && typeof (v as any).delete === 'function' && typeof (v as any).cols === 'number'
}

/** Returns a BGR 3-channel view of any image Mat. The result is a new Mat the caller owns. */
export function toBgr(cv: any, src: any): any {
  const out = new cv.Mat()
  if (src.channels() === 1) cv.cvtColor(src, out, cv.COLOR_GRAY2BGR)
  else if (src.channels() === 4) cv.cvtColor(src, out, cv.COLOR_BGRA2BGR)
  else src.copyTo(out)
  return out
}

/** Returns a single-channel grayscale view of any image Mat. New Mat, caller owns it. */
export function toGray(cv: any, src: any): any {
  const out = new cv.Mat()
  if (src.channels() === 1) src.copyTo(out)
  else cv.cvtColor(src, out, cv.COLOR_BGR2GRAY)
  return out
}

/** Renders a Mat to a canvas, normalising channel count and depth first. */
export function matToCanvas(cv: any, mat: any): OffscreenCanvas {
  const canvas = makeCanvas(mat.cols, mat.rows)

  // 8-bit only beyond this point; float maps (distance transforms) need scaling first.
  let display = mat
  let temp: any = null
  if (mat.type() !== cv.CV_8UC1 && mat.type() !== cv.CV_8UC3 && mat.type() !== cv.CV_8UC4) {
    temp = new cv.Mat()
    cv.normalize(mat, temp, 0, 255, cv.NORM_MINMAX)
    const converted = new cv.Mat()
    temp.convertTo(converted, cv.CV_8U)
    temp.delete()
    temp = converted
    display = converted
  }

  drawMatToCanvas(cv, canvas, display)
  if (temp) temp.delete()
  return canvas
}

/** Base64 JPEG (no data: prefix) — the format the desktop engine publishes previews in. */
export async function matToBase64(cv: any, mat: any, maxWidth = 480, quality = 0.75): Promise<string> {
  const full = matToCanvas(cv, mat)
  if (full.width <= maxWidth) {
    return canvasToBase64(full, quality)
  }
  const scaledHeight = Math.max(1, Math.round((full.height * maxWidth) / full.width))
  const scaled = makeCanvas(maxWidth, scaledHeight)
  scaled.getContext('2d')!.drawImage(full, 0, 0, scaled.width, scaled.height)
  return canvasToBase64(scaled, quality)
}

/**
 * The seven Hu invariants, computed from the normalised central moments.
 *
 * The OpenCV build used by the web engine does not expose `cv.HuMoments` in its
 * JS bindings, but the values are a closed-form function of moments we already
 * have, so nothing is lost by evaluating them here.
 */
export function huMoments(moments: any): number[] {
  const { nu20, nu11, nu02, nu30, nu21, nu12, nu03 } = moments

  const a = nu30 + nu12
  const b = nu21 + nu03
  const c = nu30 - 3 * nu12
  const d = 3 * nu21 - nu03

  const h1 = nu20 + nu02
  const h2 = (nu20 - nu02) ** 2 + 4 * nu11 ** 2
  const h3 = c ** 2 + d ** 2
  const h4 = a ** 2 + b ** 2
  const h5 = c * a * (a ** 2 - 3 * b ** 2) + d * b * (3 * a ** 2 - b ** 2)
  const h6 = (nu20 - nu02) * (a ** 2 - b ** 2) + 4 * nu11 * a * b
  const h7 = d * a * (a ** 2 - 3 * b ** 2) - c * b * (3 * a ** 2 - b ** 2)

  return [h1, h2, h3, h4, h5, h6, h7]
}

/**
 * Draws a polyline through `points`, closing it when asked.
 *
 * Stands in for `cv.polylines`, which this OpenCV build omits from its JS
 * bindings. Points are pixel coordinates.
 */
export function drawPolyline(
  cv: any,
  image: any,
  points: { x: number; y: number }[],
  closed: boolean,
  colour: any,
  thickness = 1
): void {
  if (points.length < 2) return
  const last = closed ? points.length : points.length - 1
  for (let i = 0; i < last; i++) {
    const from = points[i]
    const to = points[(i + 1) % points.length]
    cv.line(
      image,
      new cv.Point(Math.round(from.x), Math.round(from.y)),
      new cv.Point(Math.round(to.x), Math.round(to.y)),
      colour,
      thickness,
      cv.LINE_AA
    )
  }
}

/** Parses "#RRGGBB" into an OpenCV BGRA Scalar. */
export function parseColor(cv: any, hex: string, fallback: [number, number, number] = [0, 255, 136]): any {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex ?? ''))
  if (!m) return new cv.Scalar(fallback[2], fallback[1], fallback[0], 255)
  const int = parseInt(m[1], 16)
  const r = (int >> 16) & 255
  const g = (int >> 8) & 255
  const b = int & 255
  return new cv.Scalar(b, g, r, 255)
}

export interface LabelStat {
  id: number
  area: number
  cx: number
  cy: number
  minX: number
  minY: number
  maxX: number
  maxY: number
}

/** One pass over a CV_32S label image, collecting area/centroid/bbox per label. Label 0 is background. */
export function computeLabelStats(labels: any): Map<number, LabelStat> {
  const data = labels.data32S as Int32Array
  const w = labels.cols
  const h = labels.rows
  const acc = new Map<number, { area: number; sx: number; sy: number; minX: number; minY: number; maxX: number; maxY: number }>()

  for (let y = 0; y < h; y++) {
    const row = y * w
    for (let x = 0; x < w; x++) {
      const label = data[row + x]
      if (label <= 0) continue
      let s = acc.get(label)
      if (!s) {
        s = { area: 0, sx: 0, sy: 0, minX: x, minY: y, maxX: x, maxY: y }
        acc.set(label, s)
      }
      s.area++
      s.sx += x
      s.sy += y
      if (x < s.minX) s.minX = x
      else if (x > s.maxX) s.maxX = x
      if (y > s.maxY) s.maxY = y
    }
  }

  const out = new Map<number, LabelStat>()
  for (const [id, s] of acc) {
    out.set(id, { id, area: s.area, cx: s.sx / s.area, cy: s.sy / s.area, minX: s.minX, minY: s.minY, maxX: s.maxX, maxY: s.maxY })
  }
  return out
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const sN = s / 100
  const lN = l / 100
  const k = (n: number) => (n + h / 30) % 12
  const a = sN * Math.min(lN, 1 - lN)
  const f = (n: number) => lN - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)))
  return [Math.round(255 * f(0)), Math.round(255 * f(8)), Math.round(255 * f(4))]
}

/** Deterministic per-label colour — neighbouring ids land far apart on the hue wheel. */
export function labelColor(id: number): [number, number, number] {
  const hue = (((id * 2654435761) % 360) + 360) % 360
  return hslToRgb(hue, 65, 55)
}

/** Renders a CV_32S label image as a BGR preview. New Mat, caller owns it. */
export function colorizeLabels(cv: any, labels: any): any {
  const w = labels.cols
  const h = labels.rows
  const src = labels.data32S as Int32Array
  const out = new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(0, 0, 0))
  const dst = out.data as Uint8Array
  const cache = new Map<number, [number, number, number]>()

  for (let i = 0; i < w * h; i++) {
    const label = src[i]
    if (label <= 0) continue
    let rgb = cache.get(label)
    if (!rgb) {
      rgb = labelColor(label)
      cache.set(label, rgb)
    }
    const off = i * 3
    dst[off] = rgb[2]
    dst[off + 1] = rgb[1]
    dst[off + 2] = rgb[0]
  }
  return out
}
