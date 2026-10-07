/** Shared OpenCV.js helpers used across the browser node implementations. */
import { makeCanvas, canvasToBase64, drawMatToCanvas } from './canvasCompat'
import type { RunContext } from './types'

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

/**
 * A single-channel CV_32S view of a label map, whatever depth it arrived in —
 * the `labels.astype(np.int32)` the desktop plugins start with. Reading
 * `data32S` straight off an 8-bit or float Mat does not throw: it reinterprets
 * the bytes, so a mask wired into a Label Map port yields silent garbage.
 * Returns `labels` itself when it already fits, otherwise a new Mat the caller owns.
 */
export function toLabels32S(cv: any, labels: any): any {
  if (labels.type() === cv.CV_32SC1) return labels
  const single = labels.channels() > 1 ? new cv.Mat() : labels
  if (single !== labels) cv.extractChannel(labels, single, 0)
  const out = new cv.Mat()
  single.convertTo(out, cv.CV_32S)
  if (single !== labels) single.delete()
  return out
}

/** toLabels32S for a node's input: a conversion is freed with the run, the input itself never is. */
export function inputLabels32S(ctx: RunContext, labels: any): any {
  const out = toLabels32S(ctx.cv, labels)
  return out === labels ? labels : ctx.track(out)
}

/**
 * A CV_32S label map at another size, nearest-neighbour so no id is blended.
 * cv.resize has no CV_32S path, so the ids ride through float32 — exact for
 * any id below 2²⁴. Tracked by the run, like inputLabels32S.
 */
export function resizeLabels32S(ctx: RunContext, labels: any, width: number, height: number): any {
  const cv = ctx.cv
  const asFloat = new cv.Mat()
  labels.convertTo(asFloat, cv.CV_32F)
  const resized = new cv.Mat()
  cv.resize(asFloat, resized, new cv.Size(width, height), 0, 0, cv.INTER_NEAREST)
  asFloat.delete()
  const out = ctx.track(new cv.Mat())
  resized.convertTo(out, cv.CV_32S)
  resized.delete()
  return out
}

/** Renders a Mat to a canvas, normalising channel count and depth first. */
/**
 * An 8-bit view of `mat`, scaled from its own range when it is not already one.
 *
 * Float maps (distance transforms) and 32-bit label maps have to be brought
 * into 0-255 before anything can display — or resize them, since cv.resize has
 * no CV_32S path at all and would throw on a watershed's output.
 *
 * Returns the Mat to use and, when a conversion happened, the temporary to free.
 */
function toDisplayable8(cv: any, mat: any): { display: any; temp: any } {
  if (mat.type() === cv.CV_8UC1 || mat.type() === cv.CV_8UC3 || mat.type() === cv.CV_8UC4) {
    return { display: mat, temp: null }
  }
  const normalised = new cv.Mat()
  cv.normalize(mat, normalised, 0, 255, cv.NORM_MINMAX)
  const converted = new cv.Mat()
  normalised.convertTo(converted, cv.CV_8U)
  normalised.delete()
  return { display: converted, temp: converted }
}

export function matToCanvas(cv: any, mat: any): OffscreenCanvas {
  const canvas = makeCanvas(mat.cols, mat.rows)
  const { display, temp } = toDisplayable8(cv, mat)
  drawMatToCanvas(cv, canvas, display)
  if (temp) temp.delete()
  return canvas
}

/** Base64 JPEG (no data: prefix) — the format the desktop engine publishes previews in. */
/**
 * The Mat as an ImageBitmap, downscaled to `maxWidth`, ready to be transferred.
 *
 * The main preview used to travel as a base64 JPEG: encode, base64-encode, then
 * structured-clone a string of a few hundred kilobytes, every frame. An
 * ImageBitmap is a transferable — it moves to the main thread without a copy,
 * and no encoder runs at all. The caller owns the result and must close() it.
 */
export function matToImageBitmap(cv: any, mat: any, maxWidth = 1280): ImageBitmap {
  const { display, temp } = toDisplayable8(cv, mat)
  let source = display
  let scaled: any = null
  if (display.cols > maxWidth) {
    const height = Math.max(1, Math.round((display.rows * maxWidth) / display.cols))
    scaled = new cv.Mat()
    cv.resize(display, scaled, new cv.Size(maxWidth, height), 0, 0, cv.INTER_AREA)
    source = scaled
  }
  try {
    const canvas = makeCanvas(source.cols, source.rows)
    drawMatToCanvas(cv, canvas, source)
    // Hands the pixels over and leaves the canvas blank; nothing is copied.
    return canvas.transferToImageBitmap()
  } finally {
    if (scaled) scaled.delete()
    if (temp) temp.delete()
  }
}

export async function matToBase64(cv: any, mat: any, maxWidth = 480, quality = 0.75): Promise<string> {
  // Downscale in WASM rather than on a canvas. The obvious version paints the
  // Mat at full size and then blits it into a second, smaller canvas — two
  // allocations and a scaling blit per thumbnail, which measured at roughly
  // twice the cost of the JPEG encode itself. cv.resize on the Mat means one
  // canvas, at the size we actually want.
  if (mat.cols <= maxWidth) {
    return canvasToBase64(matToCanvas(cv, mat), quality)
  }
  // The 8-bit conversion comes first: resize has no CV_32S path, so a label map
  // would throw if it were shrunk before being brought into range.
  const { display, temp } = toDisplayable8(cv, mat)
  const height = Math.max(1, Math.round((display.rows * maxWidth) / display.cols))
  const small = new cv.Mat()
  cv.resize(display, small, new cv.Size(maxWidth, height), 0, 0, cv.INTER_AREA)
  try {
    const canvas = makeCanvas(small.cols, small.rows)
    drawMatToCanvas(cv, canvas, small)
    return await canvasToBase64(canvas, quality)
  } finally {
    small.delete()
    if (temp) temp.delete()
  }
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

/** OpenCV.js's binding omits arrowedLine — draws the shaft plus a two-stroke arrowhead instead. */
export function drawArrowedLine(cv: any, image: any, from: any, to: any, colour: any, thickness = 1, tipLength = 0.15): void {
  cv.line(image, from, to, colour, thickness, cv.LINE_AA)
  const dx = to.x - from.x
  const dy = to.y - from.y
  const length = Math.hypot(dx, dy) || 1
  const angle = Math.atan2(dy, dx)
  const tip = length * tipLength
  const spread = Math.PI / 6
  const p1 = new cv.Point(Math.round(to.x - tip * Math.cos(angle - spread)), Math.round(to.y - tip * Math.sin(angle - spread)))
  const p2 = new cv.Point(Math.round(to.x - tip * Math.cos(angle + spread)), Math.round(to.y - tip * Math.sin(angle + spread)))
  cv.line(image, to, p1, colour, thickness, cv.LINE_AA)
  cv.line(image, to, p2, colour, thickness, cv.LINE_AA)
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

/**
 * `Math.max` and `Math.min` over an array of any size.
 *
 * `Math.max(...values)` passes every element as a separate argument, so a
 * megapixel array overflows the call stack — "Maximum call stack size
 * exceeded", thrown from a line that reads like plain arithmetic. It only shows
 * up once something feeds these a full image, which is exactly what a
 * georeferenced raster does.
 */
export function maxOf(values: ArrayLike<number>, initial = -Infinity): number {
  let best = initial
  for (let i = 0; i < values.length; i++) if (values[i] > best) best = values[i]
  return best
}

export function minOf(values: ArrayLike<number>, initial = Infinity): number {
  let best = initial
  for (let i = 0; i < values.length; i++) if (values[i] < best) best = values[i]
  return best
}
