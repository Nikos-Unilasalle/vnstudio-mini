/**
 * Nodes driven by something the user draws on the node itself: Manual Points,
 * the Ruler and the Index Painter, plus the PBR Material Generator.
 *
 * The interactive ones take their strokes from a JSON string parameter that the
 * UI writes, so the node itself is pure rendering — the same contract the
 * desktop uses, which is why the visual size gate ported the same way.
 */
import type { NodeImpl } from '../types'
import { applyColormap, infernoColor, jetColor, plasmaColor, turboColor, viridisColor } from '../colormaps'

/** HERSHEY_SIMPLEX metrics, standing in for the absent `cv.getTextSize`. */
function textSize(text: string, scale: number): { width: number; height: number } {
  return { width: Math.round(17 * scale * text.length), height: Math.round(22 * scale) }
}

function parseJsonArray(raw: unknown): unknown[] {
  try {
    const parsed = JSON.parse(String(raw ?? '[]') || '[]')
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

/** 8-bit BGR copy of any input Mat, which every node here draws on. */
function toBgr8(cv: any, image: any): any {
  const eight = new cv.Mat()
  if (image.depth() !== cv.CV_8U) image.convertTo(eight, cv.CV_8U, 255)
  else image.copyTo(eight)
  const out = new cv.Mat()
  if (eight.channels() === 1) cv.cvtColor(eight, out, cv.COLOR_GRAY2BGR)
  else if (eight.channels() === 4) cv.cvtColor(eight, out, cv.COLOR_BGRA2BGR)
  else eight.copyTo(out)
  eight.delete()
  return out
}

/* ------------------------------------------------------------ manual points */

export const manualPoints: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = (inputs.image ?? inputs.main) as any
  if (!image || typeof image.cols !== 'number') return { main: null, points: [], count: 0 }

  const w = image.cols
  const h = image.rows
  const radius = Math.max(1, Math.round(Number(params.point_radius ?? 8)))
  const showLabels = params.show_labels !== false
  const annotated = ctx.track(toBgr8(cv, image))
  const font = cv.FONT_HERSHEY_SIMPLEX

  const raw = parseJsonArray(params.points)
  const points: { x: number; y: number; label: number }[] = []
  raw.forEach((entry, i) => {
    if (!entry || typeof entry !== 'object') return
    const point = entry as Record<string, unknown>
    if (!('x' in point) || !('y' in point)) return
    const x = Number(point.x)
    const y = Number(point.y)
    const label = Math.round(Number(point.label ?? 1))
    points.push({ x, y, label })

    // Stored normalised, drawn in pixels.
    const cx = Math.trunc(x * w)
    const cy = Math.trunc(y * h)
    // Foreground green, background red, as the SAM prompt convention wants.
    const colour = label === 1 ? new cv.Scalar(80, 220, 0, 255) : new cv.Scalar(255, 60, 60, 255)
    const outline = label === 1 ? new cv.Scalar(255, 255, 255, 255) : new cv.Scalar(200, 200, 200, 255)
    cv.circle(annotated, new cv.Point(cx, cy), radius, colour, -1)
    cv.circle(annotated, new cv.Point(cx, cy), radius + 1, outline, 2)
    if (showLabels) {
      cv.putText(annotated, String(i + 1), new cv.Point(cx + radius + 4, cy + 4), font, 0.4, outline, 1, cv.LINE_AA)
    }
  })

  if (points.length > 0) {
    const foreground = points.filter((p) => p.label === 1).length
    cv.putText(annotated, `FG:${foreground}  BG:${points.length - foreground}`, new cv.Point(10, h - 10), font, 0.5,
      new cv.Scalar(255, 255, 255, 255), 1, cv.LINE_AA)
  }

  return { main: annotated, points, count: points.length }
}

/* -------------------------------------------------------------------- ruler */

export const sciVisualMeasure: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = (inputs.image ?? inputs.main) as any
  if (!image || typeof image.cols !== 'number') return { main: null, length: 0, angle: 0 }

  const w = image.cols
  const h = image.rows
  const preview = ctx.track(toBgr8(cv, image))
  const font = cv.FONT_HERSHEY_SIMPLEX
  const cyan = new cv.Scalar(255, 230, 0, 255)

  // With no calibration wired the measurement stays in pixels.
  const wiredFactor = inputs.factor
  let factor = Number(wiredFactor)
  let unit: string
  if (wiredFactor === undefined || wiredFactor === null || !Number.isFinite(factor) || factor <= 0) {
    factor = 1
    unit = 'px'
  } else {
    unit = inputs.unit !== undefined && inputs.unit !== null ? String(inputs.unit) : String(params.unit ?? 'px')
  }

  const points = parseJsonArray(params.points) as Record<string, unknown>[]
  let length = 0
  let angle = 0

  if (points.length >= 2) {
    const at = (i: number): [number, number] => [Number(points[i].x) * w, Number(points[i].y) * h]
    const p1 = at(0)
    const p2 = at(1)
    const dx1 = p2[0] - p1[0]
    const dy1 = p2[1] - p1[1]
    const first = Math.hypot(dx1, dy1)
    const round = (p: [number, number]) => new cv.Point(Math.round(p[0]), Math.round(p[1]))

    let label: string
    let markX: number
    let labelY: number

    if (points.length >= 3) {
      // Three points measure a path and the interior angle at the middle one.
      const p3 = at(2)
      const second = Math.hypot(p3[0] - p2[0], p3[1] - p2[1])
      length = (first + second) / factor
      const v1x = p1[0] - p2[0]
      const v1y = p1[1] - p2[1]
      const v2x = p3[0] - p2[0]
      const v2y = p3[1] - p2[1]
      angle = (Math.atan2(Math.abs(v1x * v2y - v1y * v2x), v1x * v2x + v1y * v2y) * 180) / Math.PI

      cv.line(preview, round(p1), round(p2), cyan, 2, cv.LINE_AA)
      cv.line(preview, round(p2), round(p3), cyan, 2, cv.LINE_AA)
      for (const p of [p1, p2, p3]) cv.circle(preview, round(p), 4, cyan, -1)
      markX = Math.round(p2[0])
      labelY = Math.max(20, Math.round(p2[1]) - 18)
      label = `L: ${length.toFixed(2)} ${unit}  A: ${angle.toFixed(1)} deg`
    } else {
      length = first / factor
      // The reference direction decides where zero degrees points.
      angle = Math.round(Number(params.angle_ref ?? 0)) === 0
        ? (Math.atan2(-dy1, dx1) * 180) / Math.PI
        : (Math.atan2(dx1, -dy1) * 180) / Math.PI
      if (angle < 0) angle += 360

      cv.line(preview, round(p1), round(p2), cyan, 2, cv.LINE_AA)
      cv.circle(preview, round(p1), 4, cyan, -1)
      cv.circle(preview, round(p2), 4, cyan, -1)
      markX = Math.trunc((p1[0] + p2[0]) / 2)
      labelY = Math.max(20, Math.trunc((p1[1] + p2[1]) / 2) - 12)
      label = `L: ${length.toFixed(2)} ${unit}`
    }

    const size = textSize(label, 0.7)
    const tx = markX - Math.trunc(size.width / 2)
    const pad = 5
    cv.rectangle(preview, new cv.Point(tx - pad, labelY - size.height - pad),
      new cv.Point(tx + size.width + pad, labelY + pad), new cv.Scalar(0, 0, 0, 255), -1)
    cv.putText(preview, label, new cv.Point(tx, labelY), font, 0.7, cyan, 2, cv.LINE_AA)
  } else {
    cv.putText(preview, 'Draw a line to measure', new cv.Point(10, 24), font, 0.5,
      new cv.Scalar(255, 180, 80, 255), 1, cv.LINE_AA)
  }

  return { main: preview, length, angle }
}

/* ------------------------------------------------------------ index painter */

/** The desktop's Red-Yellow-Green ramp, for NDVI-style maps. */
function rdYlGn(value: number): [number, number, number] {
  const t = value / 255
  if (t < 0.5) return [220, Math.trunc(440 * t), 30]
  return [Math.max(0, Math.trunc(440 * (1 - t))), 200, 30]
}

function grayRamp(value: number): [number, number, number] {
  return [value, value, value]
}

const PAINTER_COLORMAPS: ((v: number) => [number, number, number])[] = [
  rdYlGn, viridisColor, infernoColor, plasmaColor, jetColor, turboColor, grayRamp,
]

export const sciIndexPainter: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const w = Math.max(1, Math.round(Number(params.width ?? 512)))
  const h = Math.max(1, Math.round(Number(params.height ?? 512)))
  const background = Number(params.bg_value ?? 0)

  const classes = parseJsonArray(params.classes) as Record<string, unknown>[]
  const strokes = parseJsonArray(params.strokes) as Record<string, unknown>[]

  const index = ctx.track(new cv.Mat(h, w, cv.CV_32F, new cv.Scalar(background)))
  const labels = ctx.track(new cv.Mat(h, w, cv.CV_8U, new cv.Scalar(0)))

  for (const stroke of strokes) {
    const classIndex = Math.round(Number(stroke.class_idx ?? 0))
    if (classIndex >= classes.length || classIndex < 0) continue
    const value = Number(classes[classIndex]?.value ?? 0)
    const radius = Math.max(1, Math.trunc(Number(stroke.radius ?? 0.03) * Math.min(w, h)))
    const points = Array.isArray(stroke.pts) ? (stroke.pts as unknown[]) : []
    // Label 0 is background, so the classes start at 1.
    const labelValue = classIndex + 1

    points.forEach((entry, i) => {
      const pair = entry as [number, number]
      const cx = Math.trunc(Number(pair[0]) * w)
      const cy = Math.trunc(Number(pair[1]) * h)
      cv.circle(index, new cv.Point(cx, cy), radius, new cv.Scalar(value), -1)
      cv.circle(labels, new cv.Point(cx, cy), radius, new cv.Scalar(labelValue), -1)
      if (i > 0) {
        const previous = points[i - 1] as [number, number]
        const px = Math.trunc(Number(previous[0]) * w)
        const py = Math.trunc(Number(previous[1]) * h)
        // A thickness of 2r joins the dots without a scalloped edge.
        cv.line(index, new cv.Point(px, py), new cv.Point(cx, cy), new cv.Scalar(value), radius * 2)
        cv.line(labels, new cv.Point(px, py), new cv.Point(cx, cy), new cv.Scalar(labelValue), radius * 2)
      }
    })
  }

  // Preview: the index stretched to 0-255 and colour-mapped.
  const range = cv.minMaxLoc(index)
  const span = range.maxVal - range.minVal
  const norm = new cv.Mat(h, w, cv.CV_8U)
  if (span > 0) index.convertTo(norm, cv.CV_8U, 255 / span, (-255 * range.minVal) / span)
  else (norm.data as Uint8Array).fill(0)

  const choice = Math.round(Number(params.colormap ?? 0))
  const colour = PAINTER_COLORMAPS[choice] ?? rdYlGn
  const preview = ctx.track(applyColormap(cv, norm, colour))
  norm.delete()

  return { index, labels, main: preview, preview }
}

/* ---------------------------------------------------- PBR material generator */

/** Sobel-derived tangent-space normals, encoded BGR as B=Z, G=Y, R=X. */
function normalsFromFloat(cv: any, source: any, strength: number): any {
  const dx = new cv.Mat()
  const dy = new cv.Mat()
  cv.Sobel(source, dx, cv.CV_32F, 1, 0, 3)
  cv.Sobel(source, dy, cv.CV_32F, 0, 1, 3)
  const gx = dx.data32F as Float32Array
  const gy = dy.data32F as Float32Array

  const out = new cv.Mat(source.rows, source.cols, cv.CV_8UC3)
  const bytes = out.data as Uint8Array
  for (let i = 0; i < gx.length; i++) {
    const x = gx[i] * strength
    const y = gy[i] * strength
    const length = Math.sqrt(x * x + y * y + 1) + 1e-8
    bytes[i * 3] = (1 / length) * 0.5 * 255 + 127.5
    bytes[i * 3 + 1] = (y / length) * 0.5 * 255 + 127.5
    bytes[i * 3 + 2] = (x / length) * 0.5 * 255 + 127.5
  }
  dx.delete()
  dy.delete()
  return out
}

export const pbrMaterialGen: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = (inputs.image ?? inputs.main) as any
  if (!image || typeof image.cols !== 'number') return {}

  const strength = Number(params.normal_strength ?? 4)
  const roughRadius = Math.max(1, Math.round(Number(params.roughness_radius ?? 6)))
  const invertRough = Boolean(params.invert_roughness)
  const aoRadius = Math.max(1, Math.round(Number(params.ao_radius ?? 16)))

  const bgr = toBgr8(cv, image)
  const gray = new cv.Mat()
  cv.cvtColor(bgr, gray, cv.COLOR_BGR2GRAY)
  const w = bgr.cols
  const h = bgr.rows

  // Albedo: frequency separation removes the broad lighting, keeps the diffuse.
  const bgrFloat = new cv.Mat()
  bgr.convertTo(bgrFloat, cv.CV_32F)
  const lowColour = new cv.Mat()
  cv.GaussianBlur(bgrFloat, lowColour, new cv.Size(0, 0), 64, 64, cv.BORDER_DEFAULT)
  const lowMean = cv.mean(lowColour)
  const midpoint = (lowMean[0] + lowMean[1] + lowMean[2]) / 3
  const albedo = ctx.track(new cv.Mat(h, w, cv.CV_8UC3))
  {
    const source = bgrFloat.data32F as Float32Array
    const low = lowColour.data32F as Float32Array
    const target = albedo.data as Uint8Array
    for (let i = 0; i < source.length; i++) target[i] = Math.min(255, Math.max(0, source[i] - low[i] + midpoint))
  }
  bgrFloat.delete()
  lowColour.delete()

  // Roughness: local standard deviation, stretched to the full range.
  const grayFloat = new cv.Mat()
  gray.convertTo(grayFloat, cv.CV_32F)
  const squared = new cv.Mat()
  cv.multiply(grayFloat, grayFloat, squared)
  const kernel = new cv.Size(roughRadius * 2 + 1, roughRadius * 2 + 1)
  const mean = new cv.Mat()
  const meanSquared = new cv.Mat()
  cv.blur(grayFloat, mean, kernel)
  cv.blur(squared, meanSquared, kernel)
  const deviation = new cv.Mat(h, w, cv.CV_32F)
  {
    const m = mean.data32F as Float32Array
    const ms = meanSquared.data32F as Float32Array
    const d = deviation.data32F as Float32Array
    for (let i = 0; i < d.length; i++) d[i] = Math.sqrt(Math.max(ms[i] - m[i] * m[i], 0))
  }
  // The desktop normalises into a float Mat and then casts, which truncates;
  // normalising straight into CV_8U would round instead and shift half the
  // pixels by one grey level.
  const roughFloat = new cv.Mat()
  cv.normalize(deviation, roughFloat, 0, 255, cv.NORM_MINMAX, cv.CV_32F)
  const roughGray = new cv.Mat(h, w, cv.CV_8U)
  {
    const source = roughFloat.data32F as Float32Array
    const target = roughGray.data as Uint8Array
    for (let i = 0; i < source.length; i++) target[i] = Math.trunc(source[i])
  }
  roughFloat.delete()
  if (invertRough) cv.bitwise_not(roughGray, roughGray)
  const roughness = ctx.track(new cv.Mat())
  cv.cvtColor(roughGray, roughness, cv.COLOR_GRAY2BGR)
  squared.delete(); mean.delete(); meanSquared.delete(); deviation.delete(); roughGray.delete()

  // Height: the same frequency separation, then CLAHE to bring out the relief.
  const lowGray = new cv.Mat()
  cv.GaussianBlur(grayFloat, lowGray, new cv.Size(0, 0), 32, 32, cv.BORDER_DEFAULT)
  const relief = new cv.Mat(h, w, cv.CV_8U)
  {
    const g = grayFloat.data32F as Float32Array
    const low = lowGray.data32F as Float32Array
    const target = relief.data as Uint8Array
    for (let i = 0; i < g.length; i++) target[i] = Math.min(255, Math.max(0, g[i] - low[i] + 128))
  }
  const clahe = new cv.CLAHE(3.0, new cv.Size(8, 8))
  const heightGray = new cv.Mat()
  clahe.apply(relief, heightGray)
  clahe.delete()
  const height = ctx.track(new cv.Mat())
  cv.cvtColor(heightGray, height, cv.COLOR_GRAY2BGR)
  lowGray.delete(); relief.delete()

  // Normals from the (blurred) grayscale.
  const smoothed = new cv.Mat()
  const scaled = new cv.Mat()
  gray.convertTo(scaled, cv.CV_32F, 1 / 255)
  cv.GaussianBlur(scaled, smoothed, new cv.Size(0, 0), 1.5, 1.5, cv.BORDER_DEFAULT)
  const normal = ctx.track(normalsFromFloat(cv, smoothed, strength))
  smoothed.delete(); scaled.delete()

  // Ambient occlusion: a multi-scale cavity measure over the height map.
  const heightFloat = new cv.Mat()
  heightGray.convertTo(heightFloat, cv.CV_32F, 1 / 255)
  const scales = [Math.max(1, Math.trunc(aoRadius / 4)), Math.max(2, Math.trunc(aoRadius / 2)), aoRadius, aoRadius * 2]
  const weights = [0.35, 0.3, 0.2, 0.15]
  const occlusion = new Float32Array(w * h)
  const heights = heightFloat.data32F as Float32Array
  for (let s = 0; s < scales.length; s++) {
    const local = new cv.Mat()
    cv.GaussianBlur(heightFloat, local, new cv.Size(0, 0), scales[s], scales[s], cv.BORDER_DEFAULT)
    const values = local.data32F as Float32Array
    for (let i = 0; i < occlusion.length; i++) {
      occlusion[i] += Math.min(1, Math.max(0, (values[i] - heights[i]) * 4)) * weights[s]
    }
    local.delete()
  }
  const aoGray = new cv.Mat(h, w, cv.CV_8U)
  {
    const target = aoGray.data as Uint8Array
    // White is fully exposed, dark is occluded — the PBR convention.
    for (let i = 0; i < occlusion.length; i++) {
      target[i] = Math.pow(Math.min(1, Math.max(0, 1 - occlusion[i])), 0.8) * 255
    }
  }
  const ao = ctx.track(new cv.Mat())
  cv.cvtColor(aoGray, ao, cv.COLOR_GRAY2BGR)
  heightFloat.delete(); aoGray.delete(); heightGray.delete()

  bgr.delete(); gray.delete(); grayFloat.delete()

  return { main: albedo, albedo, normal, roughness, height, ao }
}
