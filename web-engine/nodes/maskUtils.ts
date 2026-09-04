import type { NodeImpl } from '../types'
import { toBgr, toGray } from '../cvUtils'
import { applyColormap, cividisColor, hotColor, infernoColor, jetColor, magmaColor, rainbowColor, viridisColor } from '../colormaps'

/** #rrggbb → BGR, the order OpenCV expects. */
function hexToBgr(raw: unknown, fallback: [number, number, number] = [0, 0, 0]): [number, number, number] {
  const hex = String(raw ?? '').trim().replace(/^#/, '')
  if (!/^[0-9a-fA-F]{6}$/.test(hex)) return fallback
  return [parseInt(hex.slice(4, 6), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(0, 2), 16)]
}

/* -------------------------------------------------------------- fill holes */

export const utilFillHoles: NodeImpl = (inputs, _params, ctx) => {
  const cv = ctx.cv
  const src = (inputs.mask ?? inputs.main) as any
  if (!src) return { mask: null }

  const mask = toGray(cv, src)

  // Flood the background inwards from a corner: whatever the flood cannot reach
  // is enclosed, so inverting the flooded copy gives exactly the holes.
  const flooded = mask.clone()
  const floodMask = cv.Mat.zeros(mask.rows + 2, mask.cols + 2, cv.CV_8U)
  cv.floodFill(flooded, floodMask, new cv.Point(0, 0), new cv.Scalar(255))
  floodMask.delete()

  const holes = new cv.Mat()
  cv.bitwise_not(flooded, holes)
  const out = ctx.track(new cv.Mat())
  cv.bitwise_or(mask, holes, out)

  flooded.delete()
  holes.delete()
  mask.delete()
  return { mask: out }
}

/* ---------------------------------------------------------------- colormap */

// Same order as the desktop's enum, mapped onto the hand-written tables that
// stand in for cv.applyColorMap (absent from this OpenCV build).
const COLORMAP_CHOICES = [jetColor, hotColor, magmaColor, viridisColor, infernoColor, cividisColor, rainbowColor]

export const utilColormap: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const src = (inputs.image ?? inputs.main) as any
  if (!src) return { image: null }

  const gray = toGray(cv, src)

  // Float maps and near-black 0–1 data are stretched to the full byte range
  // first, otherwise the colormap would only ever see its bottom end.
  let eight: any
  if (gray.depth() !== cv.CV_8U) {
    eight = new cv.Mat()
    const data = gray.depth() === cv.CV_32F ? gray.data32F : gray.data64F
    let lo = Infinity
    let hi = -Infinity
    for (let i = 0; i < data.length; i++) {
      if (data[i] < lo) lo = data[i]
      if (data[i] > hi) hi = data[i]
    }
    eight = new cv.Mat(gray.rows, gray.cols, cv.CV_8U)
    const span = hi - lo || 1
    for (let i = 0; i < data.length; i++) eight.data[i] = Math.round(((data[i] - lo) / span) * 255)
  } else {
    let hi = 0
    for (let i = 0; i < gray.data.length; i++) if (gray.data[i] > hi) hi = gray.data[i]
    if (hi <= 1) {
      eight = new cv.Mat(gray.rows, gray.cols, cv.CV_8U)
      for (let i = 0; i < gray.data.length; i++) eight.data[i] = gray.data[i] * 255
    } else {
      eight = gray.clone()
    }
  }
  gray.delete()

  const index = Math.min(COLORMAP_CHOICES.length - 1, Math.max(0, Math.round(Number(params.map) || 0)))
  const out = ctx.track(applyColormap(cv, eight, COLORMAP_CHOICES[index]))
  eight.delete()
  return { image: out }
}

/* -------------------------------------------------------------- image math */

export const utilImageMath: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const src = (inputs.image ?? inputs.main) as any
  if (!src) return { image: null }

  const power = Number(params.power) ?? 1
  const out = ctx.track(new cv.Mat())
  src.convertTo(out, src.type())

  // Gamma on the 0–1 range: a 256-entry table is enough since the input is 8-bit.
  const lut = new Uint8Array(256)
  for (let v = 0; v < 256; v++) lut[v] = Math.min(255, Math.round(Math.pow(v / 255, power) * 255))
  const data = out.data
  for (let i = 0; i < data.length; i++) data[i] = lut[data[i]]

  return { image: out }
}

/* ---------------------------------------------------------- draw contours */

export const utilDrawContours: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = (inputs.image ?? inputs.main) as any
  const contours = inputs.contours as unknown[] | undefined
  if (!Array.isArray(contours)) return { image: image ?? null }

  const w = image ? image.cols : 640
  const h = image ? image.rows : 480

  const canvas = ctx.track(
    params.background || !image ? new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 255)) : toBgr(cv, image)
  )

  const [b, g, r] = hexToBgr(params.color, [0, 255, 0])
  const colour = new cv.Scalar(b, g, r, 255)
  const thickness = Math.round(Number(params.thickness) ?? 2)

  const vector = new cv.MatVector()
  let drawn = 0
  for (const entry of contours) {
    const dict = entry as { pts?: unknown[]; relative?: boolean } | null
    if (!dict || !Array.isArray(dict.pts) || dict.pts.length === 0) continue
    const relative = dict.relative !== false
    const flat: number[] = []
    for (const raw of dict.pts) {
      const p = raw as number[]
      if (!Array.isArray(p) || p.length < 2) continue
      flat.push(Math.trunc(relative ? p[0] * w : p[0]), Math.trunc(relative ? p[1] * h : p[1]))
    }
    if (flat.length < 4) continue
    vector.push_back(cv.matFromArray(flat.length / 2, 1, cv.CV_32SC2, flat))
    drawn++
  }
  if (drawn > 0) cv.drawContours(canvas, vector, -1, colour, thickness, cv.LINE_8)
  vector.delete()

  return { image: canvas }
}

/* --------------------------------------------------------- area filtering */

export const utilLabelFilterArea: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const labels = inputs.labels as any
  if (!labels) return { mask: null }

  const w = labels.cols
  const h = labels.rows
  const totalPx = w * h
  const minArea = Math.max(
    Number(params.min_area_px) || 0,
    ((Number(params.min_area_pct) ?? 0.1) / 100) * totalPx
  )
  const keepMatches = Math.round(Number(params.mode) || 0) === 0

  const data = labels.data32S ?? labels.data
  const counts = new Map<number, number>()
  for (let i = 0; i < totalPx; i++) {
    const label = data[i]
    // Watershed writes -1 on boundaries and 0 for background; neither is a region.
    if (label <= 0) continue
    counts.set(label, (counts.get(label) ?? 0) + 1)
  }

  const keep = new Set<number>()
  for (const [label, count] of counts) {
    if ((count >= minArea) === keepMatches) keep.add(label)
  }

  const out = ctx.track(new cv.Mat(h, w, cv.CV_8U))
  const bits = out.data
  for (let i = 0; i < totalPx; i++) bits[i] = keep.has(data[i]) ? 255 : 0
  return { mask: out }
}

export const maskFilterArea: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const src = (inputs.mask ?? inputs.main) as any
  if (!src) return { mask: null }

  let mask = toGray(cv, src)
  const w = mask.cols
  const h = mask.rows
  const totalPx = w * h

  if (params.invert) {
    const inverted = new cv.Mat()
    cv.bitwise_not(mask, inverted)
    mask.delete()
    mask = inverted
  }

  if (params.fill_holes) {
    // Close first so a nearly-closed outline becomes a fillable contour.
    const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5))
    const closed = new cv.Mat()
    cv.morphologyEx(mask, closed, cv.MORPH_CLOSE, kernel)
    kernel.delete()

    const contours = new cv.MatVector()
    const hierarchy = new cv.Mat()
    cv.findContours(closed, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    const filled = cv.Mat.zeros(h, w, cv.CV_8U)
    cv.drawContours(filled, contours, -1, new cv.Scalar(255), -1)
    contours.delete()
    hierarchy.delete()
    closed.delete()
    mask.delete()
    mask = filled
  }

  const minArea = Math.max(
    Number(params.min_area_px) || 0,
    ((Number(params.min_area_pct) ?? 0.1) / 100) * totalPx
  )
  const keepMatches = Math.round(Number(params.mode) || 0) === 0

  const labels = new cv.Mat()
  const stats = new cv.Mat()
  const centroids = new cv.Mat()
  const count = cv.connectedComponentsWithStats(mask, labels, stats, centroids)

  const keep = new Set<number>()
  for (let i = 1; i < count; i++) {
    const area = stats.intAt(i, cv.CC_STAT_AREA)
    if ((area >= minArea) === keepMatches) keep.add(i)
  }

  const out = ctx.track(new cv.Mat(h, w, cv.CV_8U))
  const bits = out.data
  const labelData = labels.data32S
  for (let i = 0; i < totalPx; i++) bits[i] = keep.has(labelData[i]) ? 255 : 0

  labels.delete()
  stats.delete()
  centroids.delete()
  mask.delete()
  return { mask: out }
}

/* -------------------------------------------------------- masking / outline */

export const utilImageMasking: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = (inputs.image ?? inputs.main) as any
  if (!image) return { main: null }
  const maskIn = inputs.mask as any
  if (!maskIn) return { main: image }

  let mask = toGray(cv, maskIn)
  if (mask.cols !== image.cols || mask.rows !== image.rows) {
    const resized = new cv.Mat()
    cv.resize(mask, resized, new cv.Size(image.cols, image.rows), 0, 0, cv.INTER_NEAREST)
    mask.delete()
    mask = resized
  }

  const [b, g, r] = hexToBgr(params.bg_color, [0, 0, 0])
  const colour = toBgr(cv, image)
  const out = ctx.track(new cv.Mat(image.rows, image.cols, cv.CV_8UC3, new cv.Scalar(b, g, r, 255)))
  const dst = out.data
  const srcData = colour.data
  const bits = mask.data
  for (let p = 0, i = 0; p < bits.length; p++, i += 3) {
    if (!bits[p]) continue
    dst[i] = srcData[i]
    dst[i + 1] = srcData[i + 1]
    dst[i + 2] = srcData[i + 2]
  }

  colour.delete()
  mask.delete()
  return { main: out }
}

export const maskOutline: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const src = (inputs.mask ?? inputs.main) as any
  if (!src) return { mask: null }

  const mask = toGray(cv, src)
  let thickness = Math.max(1, Math.round(Number(params.thickness) || 2))
  // A morphological gradient needs an odd kernel to stay centred on the border.
  if (thickness % 2 === 0) thickness += 1

  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(thickness, thickness))
  const out = ctx.track(new cv.Mat())
  cv.morphologyEx(mask, out, cv.MORPH_GRADIENT, kernel)
  kernel.delete()
  mask.delete()
  return { mask: out }
}

/* --------------------------------------------------------- band / halves */

/** Zeroes everything outside [from, to) along the chosen axis. */
function keepBand(cv: any, img: any, from: number, to: number, vertical: boolean): any {
  const out = cv.Mat.zeros(img.rows, img.cols, img.type())
  if (to <= from) return out
  const rect = vertical
    ? new cv.Rect(from, 0, to - from, img.rows)
    : new cv.Rect(0, from, img.cols, to - from)
  img.roi(rect).copyTo(out.roi(rect))
  return out
}

export const utilMaskBand: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = inputs.image as any
  const mask = inputs.mask as any
  const src = image ?? mask
  if (!src) return {}

  const vertical = Math.round(Number(params.axis) || 0) === 1
  const extent = vertical ? src.cols : src.rows
  const start = Number(params.start_pct) || 0
  const end = Number(params.end_pct) ?? 50
  const from = Math.max(0, Math.trunc((extent * start) / 100))
  const to = Math.max(from, Math.min(extent, Math.trunc((extent * end) / 100)))

  return {
    image: image ? ctx.track(keepBand(cv, image, from, to, vertical)) : null,
    mask: mask ? ctx.track(keepBand(cv, mask, from, to, vertical)) : null,
  }
}

export const utilSplitHalf: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = inputs.image as any
  const mask = inputs.mask as any
  const src = image ?? mask
  if (!src) return {}

  // Default axis is vertical (left / right), matching the desktop.
  const vertical = Math.round(Number(params.axis) ?? 1) === 1
  const extent = vertical ? src.cols : src.rows
  const split = Math.max(0, Math.min(extent, Math.trunc((extent * (Number(params.position) ?? 50)) / 100)))

  // Each half keeps its own position in the frame, so the two can be compared
  // pixel for pixel — which is the point for asymmetry analysis.
  const first = (img: any) => ctx.track(keepBand(cv, img, 0, split, vertical))
  const second = (img: any) => ctx.track(keepBand(cv, img, split, extent, vertical))

  return {
    first_image: image ? first(image) : null,
    second_image: image ? second(image) : null,
    first_mask: mask ? first(mask) : null,
    second_mask: mask ? second(mask) : null,
  }
}

/* ------------------------------------------------------------------ compose */

export const utilCompose: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const rawA = (inputs.image_a ?? inputs.image) as any
  const rawB = inputs.image_b as any
  if (!rawA) return { main: rawB ?? null }
  if (!rawB) return { main: rawA }

  const a = toBgr(cv, rawA)
  const b = toBgr(cv, rawB)
  const ha = a.rows
  const wa = a.cols

  const mode = Math.round(Number(params.mode) || 0)
  const split = Math.round(Number(params.split_pos) || 50)
  const gap = Math.max(0, Math.round(Number(params.gap) ?? 2))
  const alpha = Number(params.alpha) ?? 0.5

  /** B scaled to A's frame — every overlay mode needs this. */
  const fitted = () => {
    const out = new cv.Mat()
    cv.resize(b, out, new cv.Size(wa, ha), 0, 0, cv.INTER_LINEAR)
    return out
  }

  let out: any
  if (mode === 0 || mode === 1) {
    // Side by side / stacked: B keeps its aspect ratio against A's matching edge.
    const horizontal = mode === 0
    const bw = horizontal ? (b.rows > 0 ? Math.trunc((b.cols * ha) / b.rows) : b.cols) : wa
    const bh = horizontal ? ha : b.cols > 0 ? Math.trunc((b.rows * wa) / b.cols) : b.rows
    const scaled = new cv.Mat()
    cv.resize(b, scaled, new cv.Size(bw, bh), 0, 0, cv.INTER_LINEAR)

    const width = horizontal ? wa + gap + bw : wa
    const height = horizontal ? ha : ha + gap + bh
    out = ctx.track(new cv.Mat(height, width, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 255)))
    a.copyTo(out.roi(new cv.Rect(0, 0, wa, ha)))
    scaled.copyTo(out.roi(horizontal ? new cv.Rect(wa + gap, 0, bw, bh) : new cv.Rect(0, ha + gap, bw, bh)))
    scaled.delete()
  } else if (mode === 2 || mode === 3) {
    // Split view: A on one side of the line, B on the other.
    const horizontal = mode === 2
    const scaled = fitted()
    out = ctx.track(a.clone())
    const extent = horizontal ? wa : ha
    const at = Math.max(0, Math.min(extent - 1, Math.trunc((extent * split) / 100)))
    const rest = horizontal ? new cv.Rect(at, 0, wa - at, ha) : new cv.Rect(0, at, wa, ha - at)
    scaled.roi(rest).copyTo(out.roi(rest))
    if (gap > 0) {
      const thick = Math.max(1, gap)
      const from = Math.max(0, at - (thick >> 1))
      const to = Math.min(extent, from + thick)
      const line = horizontal ? new cv.Rect(from, 0, to - from, ha) : new cv.Rect(0, from, wa, to - from)
      out.roi(line).setTo(new cv.Scalar(220, 220, 220, 255))
    }
    scaled.delete()
  } else if (mode === 4) {
    const scaled = fitted()
    out = ctx.track(new cv.Mat())
    cv.addWeighted(a, alpha, scaled, 1 - alpha, 0, out)
    scaled.delete()
  } else if (mode === 5) {
    const scaled = fitted()
    const diff = new cv.Mat()
    cv.absdiff(a, scaled, diff)
    out = ctx.track(new cv.Mat())
    // Doubled, because a raw difference of two similar frames is nearly black.
    cv.convertScaleAbs(diff, out, 2.0, 0)
    diff.delete()
    scaled.delete()
  } else if (mode === 6) {
    const scaled = fitted()
    out = ctx.track(a.clone())
    const tile = Math.max(4, Math.trunc((Math.min(wa, ha) * split) / 100))
    for (let y = 0; y < ha; y += tile) {
      for (let x = 0; x < wa; x += tile) {
        if ((Math.floor(y / tile) + Math.floor(x / tile)) % 2 !== 1) continue
        const rect = new cv.Rect(x, y, Math.min(tile, wa - x), Math.min(tile, ha - y))
        scaled.roi(rect).copyTo(out.roi(rect))
      }
    }
    scaled.delete()
  } else {
    out = ctx.track(a.clone())
  }

  a.delete()
  b.delete()
  return { main: out }
}

/* ----------------------------------------------------------- small helpers */

export const pluginImageToMask: NodeImpl = (inputs, _params, ctx) => {
  const src = (inputs.image ?? inputs.main) as any
  if (!src) return { mask: null }
  return { mask: ctx.track(toGray(ctx.cv, src)) }
}

export const pluginInvertMask: NodeImpl = (inputs, _params, ctx) => {
  const cv = ctx.cv
  const src = (inputs.mask ?? inputs.main) as any
  if (!src) return { main: null }

  const out = ctx.track(new cv.Mat())
  if (src.depth() === cv.CV_32F || src.depth() === cv.CV_64F) {
    // Float masks run 0–1, so inverting means 1 − v, not a bitwise complement.
    src.convertTo(out, src.type())
    const data = out.depth() === cv.CV_32F ? out.data32F : out.data64F
    for (let i = 0; i < data.length; i++) data[i] = 1 - data[i]
  } else {
    cv.bitwise_not(src, out)
  }
  return { main: out }
}

/** Reads an (x, y) from a dict or a pair, deciding pixel vs normalised by range. */
function pointToPixels(entry: unknown, w: number, h: number): [number, number] | null {
  let x: number | undefined
  let y: number | undefined
  if (Array.isArray(entry) && entry.length >= 2) {
    x = Number(entry[0])
    y = Number(entry[1])
  } else if (entry && typeof entry === 'object') {
    const p = entry as Record<string, unknown>
    if (typeof p.x === 'number' && typeof p.y === 'number') {
      x = p.x
      y = p.y
    }
  }
  if (x === undefined || y === undefined || !Number.isFinite(x) || !Number.isFinite(y)) return null
  // Both coordinates within the unit square means normalised, as everything
  // upstream in this app emits; anything larger is already in pixels.
  return x <= 1 && y <= 1 ? [Math.trunc(x * w), Math.trunc(y * h)] : [Math.trunc(x), Math.trunc(y)]
}

export const pointsToMask: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const reference = inputs.reference as any
  const w = reference ? reference.cols : Math.max(1, Math.round(Number(params.width) || 640))
  const h = reference ? reference.rows : Math.max(1, Math.round(Number(params.height) || 480))

  const mask = ctx.track(cv.Mat.zeros(h, w, cv.CV_8U))
  const points = inputs.points
  if (!Array.isArray(points)) return { mask }

  const radius = Math.max(1, Math.round(Number(params.radius) || 5))
  const white = new cv.Scalar(255)
  for (const entry of points) {
    const p = pointToPixels(entry, w, h)
    if (p) cv.circle(mask, new cv.Point(p[0], p[1]), radius, white, -1)
  }
  return { mask }
}

export const mergePoints: NodeImpl = (inputs, params) => {
  const raw = inputs.points
  const empty = { points: [], keypoints: [], count: 0 }
  if (!Array.isArray(raw) || raw.length === 0) return empty

  const imageWidth = Number(params.image_width) || 640
  const imageHeight = Number(params.image_height) || 480
  const threshold = Number(params.threshold) || 20

  const pixels: [number, number][] = []
  for (const entry of raw) {
    const p = entry as Record<string, unknown> | null
    if (!p || typeof p !== 'object' || typeof p.x !== 'number' || typeof p.y !== 'number') continue
    pixels.push([p.x * imageWidth, p.y * imageHeight])
  }
  if (!pixels.length) return empty

  // Greedy clustering in pixel space: the first unassigned point seeds a
  // cluster and absorbs every later point within the radius. Order-dependent
  // by design — this mirrors the desktop, which downstream results rely on.
  const assigned = new Array<boolean>(pixels.length).fill(false)
  const merged: { x: number; y: number }[] = []
  for (let i = 0; i < pixels.length; i++) {
    if (assigned[i]) continue
    assigned[i] = true
    const cluster: [number, number][] = [pixels[i]]
    for (let j = i + 1; j < pixels.length; j++) {
      if (assigned[j]) continue
      if (Math.hypot(pixels[i][0] - pixels[j][0], pixels[i][1] - pixels[j][1]) <= threshold) {
        cluster.push(pixels[j])
        assigned[j] = true
      }
    }
    const cx = cluster.reduce((s, p) => s + p[0], 0) / cluster.length
    const cy = cluster.reduce((s, p) => s + p[1], 0) / cluster.length
    merged.push({ x: cx / imageWidth, y: cy / imageHeight })
  }

  return {
    points: merged,
    keypoints: merged.map((p) => ({
      _type: 'graphics',
      shape: 'point',
      pts: [[p.x, p.y]],
      relative: true,
      color: '#00ff88',
    })),
    count: merged.length,
  }
}
