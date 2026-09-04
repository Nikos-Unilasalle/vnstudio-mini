import type { NodeImpl } from '../types'
import { downloadFile } from '../../shims/vfs'
import { toBgr, toGray } from '../cvUtils'
import { applyColormap, jetColor, oceanColor, viridisColor } from '../colormaps'

/** #rrggbb → BGR. */
function hexToBgr(raw: unknown, fallback: [number, number, number]): [number, number, number] {
  let hex = String(raw ?? '').trim().replace(/^#/, '')
  if (hex.length === 3) hex = hex.split('').map((c) => c + c).join('')
  if (!/^[0-9a-fA-F]{6}$/.test(hex)) return fallback
  return [parseInt(hex.slice(4, 6), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(0, 2), 16)]
}

/**
 * HERSHEY_SIMPLEX text metrics. cv.getTextSize is missing from this build, and
 * the advance is close to 17px per character at scale 1 with a cap height near
 * 22px, which is what the layout below needs.
 */
function textSize(text: string, scale: number): { width: number; height: number } {
  return { width: Math.round(17 * scale * text.length), height: Math.round(22 * scale) }
}

/* ------------------------------------------------- normalised difference */

/**
 * A normalised difference index of two bands, (A−B)/(A+B). The epsilon keeps
 * the ratio finite where both bands are zero.
 */
function normalisedDifference(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length)
  for (let i = 0; i < a.length; i++) out[i] = (a[i] - b[i]) / (a[i] + b[i] + 1e-6)
  return out
}

/** Renders an index in [-1, 1] as a colour map, plus the mask above a threshold. */
function renderIndex(cv: any, ctx: any, index: Float32Array, w: number, h: number, threshold: number, colour: (v: number) => [number, number, number]) {
  const eight = new cv.Mat(h, w, cv.CV_8U)
  for (let i = 0; i < index.length; i++) {
    eight.data[i] = Math.max(0, Math.min(255, Math.round(((index[i] + 1) / 2) * 255)))
  }
  const main = ctx.track(applyColormap(cv, eight, colour))
  eight.delete()

  const mask = ctx.track(new cv.Mat(h, w, cv.CV_8U))
  let above = 0
  for (let i = 0; i < index.length; i++) {
    const on = index[i] > threshold
    mask.data[i] = on ? 255 : 0
    if (on) above++
  }
  return { main, mask, coverage: (above / index.length) * 100 }
}

const INDEX_COLOURS = [viridisColor, oceanColor, jetColor]

export const featNdwi: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const source = (inputs.image ?? inputs.main) as any
  if (!source) return { main: null, mask: null, coverage: 0 }

  const img = toBgr(cv, source)
  const w = img.cols
  const h = img.rows
  const n = w * h
  const data = img.data

  const green = new Float32Array(n)
  const other = new Float32Array(n)
  const mode = Math.round(Number(params.mode) || 0)
  const nir = inputs.nir as any

  if (mode === 1 && nir) {
    // The standard index: (Green − NIR)/(Green + NIR), needing a real NIR band.
    let nirGray = toGray(cv, nir)
    if (nirGray.cols !== w || nirGray.rows !== h) {
      const resized = new cv.Mat()
      cv.resize(nirGray, resized, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR)
      nirGray.delete()
      nirGray = resized
    }
    for (let i = 0; i < n; i++) {
      green[i] = data[i * 3 + 1]
      other[i] = nirGray.data[i]
    }
    nirGray.delete()
  } else {
    // Without a NIR band, (Blue − Green)/(Blue + Green) is the usual RGB proxy.
    for (let i = 0; i < n; i++) {
      green[i] = data[i * 3]
      other[i] = data[i * 3 + 1]
    }
  }
  img.delete()

  const index = normalisedDifference(green, other)
  const colour = INDEX_COLOURS[Math.min(2, Math.max(0, Math.round(Number(params.colormap) || 0)))]
  return renderIndex(cv, ctx, index, w, h, Number(params.threshold) ?? 0.2, colour)
}

export const featSpectralIndex: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const sourceA = (inputs.image_a ?? inputs.image ?? inputs.main) as any
  if (!sourceA) return { main: null, mask: null, coverage: 0 }

  const a = toBgr(cv, sourceA)
  const w = a.cols
  const h = a.rows
  const n = w * h

  // Presets pick the channel pair; anything else uses the manual selection.
  const presets: Record<number, [number, number]> = { 1: [0, 1], 2: [1, 2], 3: [0, 2] }
  const preset = Math.round(Number(params.preset) || 0)
  const [chA, chB] = presets[preset] ?? [Math.round(Number(params.ch_a) || 0), Math.round(Number(params.ch_b) ?? 1)]

  const extract = (mat: any, channel: number) => {
    const out = new Float32Array(mat.cols * mat.rows)
    if (mat.channels() === 1) {
      for (let i = 0; i < out.length; i++) out[i] = mat.data[i]
    } else if (channel <= 2) {
      for (let i = 0; i < out.length; i++) out[i] = mat.data[i * mat.channels() + channel]
    } else {
      // "Luminance" is the fourth option.
      const gray = toGray(cv, mat)
      for (let i = 0; i < out.length; i++) out[i] = gray.data[i]
      gray.delete()
    }
    return out
  }

  const bandA = extract(a, chA)
  let sourceB = inputs.image_b as any
  let bandB: Float32Array
  if (sourceB) {
    let b = toBgr(cv, sourceB)
    if (b.cols !== w || b.rows !== h) {
      const resized = new cv.Mat()
      cv.resize(b, resized, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR)
      b.delete()
      b = resized
    }
    bandB = extract(b, chB)
    b.delete()
  } else {
    bandB = extract(a, chB)
  }
  a.delete()

  void n
  const index = normalisedDifference(bandA, bandB)
  const colour = INDEX_COLOURS[Math.min(2, Math.max(0, Math.round(Number(params.colormap) || 0)))]
  return renderIndex(cv, ctx, index, w, h, Number(params.threshold) ?? 0.2, colour)
}

export const featWaterRefine: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const source = (inputs.mask ?? inputs.image ?? inputs.main) as any
  if (!source) return { mask: null, main: null, contours: [], count: 0 }

  const gray = toGray(cv, source)
  let binary = new cv.Mat()
  cv.threshold(gray, binary, 127, 255, cv.THRESH_BINARY)
  gray.delete()

  // Closing first fills the gaps inside a water body, then opening removes the
  // speckle; doing it the other way round would erase thin channels.
  let closeSize = Math.max(1, Math.round(Number(params.close_size) ?? 7))
  if (closeSize > 1) {
    if (closeSize % 2 === 0) closeSize += 1
    const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(closeSize, closeSize))
    const closed = new cv.Mat()
    cv.morphologyEx(binary, closed, cv.MORPH_CLOSE, kernel)
    kernel.delete()
    binary.delete()
    binary = closed
  }
  let openSize = Math.max(1, Math.round(Number(params.open_size) ?? 3))
  if (openSize > 1) {
    if (openSize % 2 === 0) openSize += 1
    const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(openSize, openSize))
    const opened = new cv.Mat()
    cv.morphologyEx(binary, opened, cv.MORPH_OPEN, kernel)
    kernel.delete()
    binary.delete()
    binary = opened
  }

  const w = binary.cols
  const h = binary.rows
  const minArea = Math.round(Number(params.min_area) ?? 500)

  const contours = new cv.MatVector()
  const hierarchy = new cv.Mat()
  cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  hierarchy.delete()

  const clean = ctx.track(cv.Mat.zeros(h, w, cv.CV_8U))
  const kept = new cv.MatVector()
  const polygons: Record<string, unknown>[] = []
  for (let i = 0; i < contours.size(); i++) {
    const contour = contours.get(i)
    if (cv.contourArea(contour) < minArea) continue
    kept.push_back(contour)
    // Long coastlines are decimated to about 60 vertices so the polygon stays
    // light enough to travel to the UI and be drawn.
    const step = Math.max(1, Math.floor(contour.rows / 60))
    const pts: number[][] = []
    for (let p = 0; p < contour.rows; p += step) {
      pts.push([contour.data32S[p * 2] / w, contour.data32S[p * 2 + 1] / h])
    }
    polygons.push({ _type: 'graphics', shape: 'polygon', pts, relative: true, color: '#00aaff', thickness: 2 })
  }
  if (kept.size() > 0) cv.drawContours(clean, kept, -1, new cv.Scalar(255), -1)

  const vis = ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 255)))
  for (let i = 0; i < clean.data.length; i++) {
    if (!clean.data[i]) continue
    vis.data[i * 3] = 200
    vis.data[i * 3 + 1] = 100
    vis.data[i * 3 + 2] = 0
  }

  const count = kept.size()
  contours.delete()
  kept.delete()
  binary.delete()

  ctx.emit('count', count)
  return { mask: clean, main: vis, contours: polygons, count }
}

/* ------------------------------------------------------------ SVG export */

export const maskToSvg: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const source = (inputs.mask ?? inputs.main) as any
  if (!source) return { mask: null }

  const state = (ctx.state.get(ctx.nodeId) as { last: number } | undefined) ?? { last: 0 }
  const trigger = Number(params.trigger) ? 1 : 0
  const fired = trigger === 1 && state.last === 0
  state.last = trigger
  ctx.state.set(ctx.nodeId, state)
  if (!fired) return { mask: source }

  const gray = toGray(cv, source)
  const binary = new cv.Mat()
  cv.threshold(gray, binary, 0, 255, cv.THRESH_BINARY)
  gray.delete()

  const w = binary.cols
  const h = binary.rows
  const contours = new cv.MatVector()
  const hierarchy = new cv.Mat()
  cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  hierarchy.delete()
  binary.delete()

  const stroke = String(params.stroke_color ?? '#000000')
  const fill = String(params.fill_color ?? '#ffffff')
  const strokeWidth = Number(params.stroke_width) || 1.5

  const paths: string[] = []
  for (let i = 0; i < contours.size(); i++) {
    const contour = contours.get(i)
    if (contour.rows < 2) continue
    const data = contour.data32S
    let d = `M ${data[0]},${data[1]}`
    for (let p = 1; p < contour.rows; p++) d += ` L ${data[p * 2]},${data[p * 2 + 1]}`
    d += ' Z'
    paths.push(`  <path d="${d}" stroke="${stroke}" fill="${fill}" stroke-width="${strokeWidth}"/>`)
  }
  contours.delete()

  const svg = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">`,
    ...paths,
    '</svg>',
  ].join('\n')

  const name = String(params.path ?? 'output.svg').split('/').pop() || 'output.svg'
  downloadFile(name.endsWith('.svg') ? name : `${name}.svg`, svg, 'image/svg+xml')
  ctx.emit('paths', paths.length)
  return { mask: source }
}

/* ---------------------------------------------------------------- legend */

export const imageLegend: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const source = (inputs.image ?? inputs.main) as any
  if (!source) return {}

  const canvas = ctx.track(toBgr(cv, source))

  // One "Label:#RRGGBB" per line.
  const entries: { label: string; colour: [number, number, number] }[] = []
  for (const line of String(params.entries ?? '').split('\n')) {
    const trimmed = line.trim()
    const at = trimmed.indexOf(':')
    if (at < 0) continue
    const label = trimmed.slice(0, at).trim()
    const colour = hexToBgr(trimmed.slice(at + 1).trim(), [128, 128, 128])
    if (label) entries.push({ label, colour })
  }
  if (!entries.length) return { main: canvas }

  const fontScale = Number(params.font_scale) || 0.5
  const swatch = Math.max(6, Math.round(Number(params.swatch_size) || 16))
  const bgAlpha = Number(params.bg_alpha) ?? 0.55
  const pad = Math.max(0, Math.round(Number(params.padding) ?? 8))
  const position = String(params.position ?? 'bottom-left')
  const lineGap = 4

  const sizes = entries.map((e) => textSize(e.label, fontScale))
  const maxTextWidth = Math.max(...sizes.map((s) => s.width))
  const rowHeight = Math.max(swatch, Math.max(...sizes.map((s) => s.height)))
  const boxW = pad + swatch + pad + maxTextWidth + pad
  const boxH = pad + entries.length * (rowHeight + lineGap) - lineGap + pad

  const W = canvas.cols
  const H = canvas.rows
  const x0 = Math.max(0, position.includes('right') ? W - boxW - 4 : 4)
  const y0 = Math.max(0, position.includes('bottom') ? H - boxH - 4 : 4)
  const x1 = Math.min(W, x0 + boxW)
  const y1 = Math.min(H, y0 + boxH)

  // Semi-transparent plate so the legend stays readable over any image.
  const overlay = canvas.clone()
  cv.rectangle(overlay, new cv.Point(x0, y0), new cv.Point(x1, y1), new cv.Scalar(20, 20, 20, 255), cv.FILLED)
  cv.addWeighted(overlay, bgAlpha, canvas, 1 - bgAlpha, 0, canvas)
  overlay.delete()

  entries.forEach((entry, i) => {
    const rowY = y0 + pad + i * (rowHeight + lineGap)
    const sx = x0 + pad
    cv.rectangle(canvas, new cv.Point(sx, rowY), new cv.Point(sx + swatch, rowY + swatch), new cv.Scalar(entry.colour[0], entry.colour[1], entry.colour[2], 255), cv.FILLED)
    cv.rectangle(canvas, new cv.Point(sx, rowY), new cv.Point(sx + swatch, rowY + swatch), new cv.Scalar(180, 180, 180, 255), 1)
    const ty = rowY + swatch - Math.floor((swatch - sizes[i].height) / 2)
    cv.putText(canvas, entry.label, new cv.Point(sx + swatch + pad, ty), cv.FONT_HERSHEY_SIMPLEX, fontScale, new cv.Scalar(230, 230, 230, 255), 1, cv.LINE_AA)
  })

  return { main: canvas }
}

/* ---------------------------------------------------------------- on each */

export const utilOnEach: NodeImpl = (inputs, params) => {
  const list = inputs.list_in
  const template = inputs.template as Record<string, unknown> | null | undefined
  if (!Array.isArray(list) || !template || typeof template !== 'object') return { list_out: [] }

  const textSource = Math.round(Number(params.text_source) || 0)
  const out: Record<string, unknown>[] = []

  list.forEach((raw, i) => {
    const item = raw as Record<string, any>
    if (!item || typeof item !== 'object') return

    // A position can arrive in any of the shapes the graph passes around.
    let x: number | undefined
    let y: number | undefined
    if (typeof item.x === 'number' && typeof item.y === 'number') {
      x = item.x
      y = item.y
    } else if (item.center && typeof item.center === 'object') {
      x = item.center.x
      y = item.center.y
    } else if (typeof item.xmin === 'number' && typeof item.ymin === 'number') {
      x = item.xmin + (Number(item.width) || 0) / 2
      y = item.ymin + (Number(item.height) || 0) / 2
    } else if (Array.isArray(item.pts) && item.pts.length) {
      x = item.pts[0][0]
      y = item.pts[0][1]
    } else if (Array.isArray(item.landmarks) && item.landmarks.length) {
      x = item.landmarks[0].x
      y = item.landmarks[0].y
    }
    if (x === undefined || y === undefined) return

    const graphic: Record<string, unknown> = { ...template, pts: [[x, y]], relative: true }

    if (graphic.shape === 'text' || 'text' in graphic || 'label' in graphic) {
      let text: string | null = null
      if (textSource === 1) text = String(item.id ?? i)
      else if (textSource === 2) text = String(item.label ?? 'obj')
      else if (textSource === 3) text = `${Math.round(Number(item.area) || 0)}`
      else if (textSource === 4) text = `${Math.round((Number(item.score) || 0) * 100)}%`
      else if (textSource === 5) text = `(${x.toFixed(2)}, ${y.toFixed(2)})`
      if (text !== null) {
        graphic.text = text
        graphic.label = text
      }
    }
    out.push(graphic)
  })

  return { list_out: out }
}

/* --------------------------------------------------------- 3D projection */

export const math3dToScreen: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const vec = inputs.vector as Record<string, any> | null | undefined
  const image = inputs.image as any
  if (!vec || typeof vec !== 'object') return { main: image ?? null, x: null, y: null, point: null }

  const x = Number(vec.x ?? vec.vec_x ?? 0)
  const y = Number(vec.y ?? vec.vec_y ?? 0)
  let z = Number(vec.z ?? vec.vec_z ?? 1)
  // A point on the camera plane has no projection; nudge it off.
  if (Math.abs(z) < 0.0001) z = 0.0001

  const f = Number(params.focal_length) || 1
  const scaleX = Number(params.scale_x) || 1
  const scaleY = Number(params.scale_y) || 1
  const offsetX = Number(params.offset_x) || 0
  const offsetY = Number(params.offset_y) || 0

  // Pinhole projection about the frame centre; screen y runs downwards.
  let sx = 0.5 + (f * (x / z) + offsetX) * scaleX
  let sy = 0.5 - (f * (y / z) + offsetY) * scaleY
  if (params.flip_x) sx = 1 - sx
  if (params.flip_y) sy = 1 - sy
  if (params.clamp !== false) {
    sx = Math.max(0, Math.min(1, sx))
    sy = Math.max(0, Math.min(1, sy))
  }

  const point = { _type: 'graphics', shape: 'point', pts: [[sx, sy]], relative: true, r: 0, g: 255, b: 100, thickness: 10 }

  let out: any = null
  if (image) {
    out = ctx.track(toBgr(cv, image))
    const px = Math.round(sx * out.cols)
    const py = Math.round(sy * out.rows)
    const colour = new cv.Scalar(100, 255, 0, 255)
    cv.circle(out, new cv.Point(px, py), 12, colour, 2, cv.LINE_AA)
    cv.line(out, new cv.Point(px - 10, py), new cv.Point(px + 10, py), colour, 1, cv.LINE_AA)
    cv.line(out, new cv.Point(px, py - 10), new cv.Point(px, py + 10), colour, 1, cv.LINE_AA)
  }

  return { main: out, x: sx, y: sy, point }
}

/* ---------------------------------------------------------- eye crops */

// MediaPipe FaceMesh indices: eye corners and iris centres.
const LEFT_INNER = 133
const LEFT_OUTER = 33
const LEFT_IRIS = 468
const RIGHT_INNER = 362
const RIGHT_OUTER = 263
const RIGHT_IRIS = 473

export const transformEyeCrop: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = (inputs.image ?? inputs.main) as any
  const face = inputs.face as { landmarks?: { x: number; y: number }[] } | null | undefined
  const size = Math.max(32, Math.round(Number(params.size) || 64))

  const blank = () => ctx.track(new cv.Mat(size, size, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 255)))
  // The iris landmarks only exist in the 478-point refined mesh.
  if (!image || !face || !Array.isArray(face.landmarks) || face.landmarks.length < 478) {
    return { eye_left: blank(), eye_right: blank(), meta: {} }
  }

  const landmarks = face.landmarks
  const padding = Number(params.padding) ?? 0.4
  const align = params.align !== false
  const source = toBgr(cv, image)
  const w = source.cols
  const h = source.rows

  const crop = (innerId: number, outerId: number, irisId: number) => {
    const inner = [landmarks[innerId].x * w, landmarks[innerId].y * h]
    const outer = [landmarks[outerId].x * w, landmarks[outerId].y * h]
    const centre = irisId < landmarks.length
      ? [landmarks[irisId].x * w, landmarks[irisId].y * h]
      : [(inner[0] + outer[0]) / 2, (inner[1] + outer[1]) / 2]

    const eyeWidth = Math.hypot(outer[0] - inner[0], outer[1] - inner[1])
    const half = Math.round(eyeWidth * (0.5 + padding))

    let angle = 0
    if (align) {
      angle = (Math.atan2(outer[1] - inner[1], outer[0] - inner[0]) * 180) / Math.PI
      // Normalised to ±90 so the left eye is not rotated a half turn.
      if (angle > 90) angle -= 180
      else if (angle < -90) angle += 180
    }

    let working = source
    let rotated: any = null
    if (align && Math.abs(angle) > 1) {
      rotated = new cv.Mat()
      const matrix = cv.getRotationMatrix2D(new cv.Point(centre[0], centre[1]), angle, 1)
      cv.warpAffine(source, rotated, matrix, new cv.Size(w, h))
      matrix.delete()
      working = rotated
    }

    const cx = Math.round(centre[0])
    const cy = Math.round(centre[1])
    const x1 = Math.max(0, cx - half)
    const y1 = Math.max(0, cy - half)
    const x2 = Math.min(w, cx + half)
    const y2 = Math.min(h, cy + half)

    let out: any
    if (x2 <= x1 || y2 <= y1) {
      out = blank()
    } else {
      const region = working.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1))
      out = ctx.track(new cv.Mat())
      cv.resize(region, out, new cv.Size(size, size), 0, 0, cv.INTER_LINEAR)
      region.delete()
    }
    rotated?.delete()

    return {
      image: out,
      meta: { cx: centre[0] / w, cy: centre[1] / h, x1: x1 / w, y1: y1 / h, x2: x2 / w, y2: y2 / h, angle },
    }
  }

  const left = crop(LEFT_INNER, LEFT_OUTER, LEFT_IRIS)
  const right = crop(RIGHT_INNER, RIGHT_OUTER, RIGHT_IRIS)
  source.delete()

  return { eye_left: left.image, eye_right: right.image, meta: { left: left.meta, right: right.meta, size } }
}
