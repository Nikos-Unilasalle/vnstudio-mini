import type { NodeImpl } from '../types'
import { drawArrowedLine, drawPolyline, parseColor, toBgr } from '../cvUtils'

interface GraphicsDict {
  _type: 'graphics'
  shape: string
  pts: [number, number][]
  relative: boolean
  color?: string
  r?: number
  g?: number
  b?: number
  thickness?: number
  fill?: boolean
  rx?: number
  ry?: number
  angle?: number
  tip_length?: number
  text?: string
  font_scale?: number
  label?: string
  radius?: number
  radius_rel?: number
}

function hexToRgb(hex: string, fallback: [number, number, number]): [number, number, number] {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex ?? ''))
  if (!m) return fallback
  const int = parseInt(m[1], 16)
  return [(int >> 16) & 255, (int >> 8) & 255, int & 255]
}

function scalarOrParam(inputs: Record<string, unknown>, params: Record<string, any>, key: string, fallback: number): number {
  const wired = inputs[key]
  if (typeof wired === 'number') return wired
  const p = params[key]
  return typeof p === 'number' ? p : fallback
}

// --- graphics-descriptor nodes (consumed by Draw Overlay) --------------------

export const drawArrow: NodeImpl = (inputs, params) => {
  const x1 = scalarOrParam(inputs, params, 'x1', 0.2)
  const y1 = scalarOrParam(inputs, params, 'y1', 0.5)
  const x2 = scalarOrParam(inputs, params, 'x2', 0.8)
  const y2 = scalarOrParam(inputs, params, 'y2', 0.5)
  const color = String(params.color ?? '#FF6600')
  const [r, g, b] = hexToRgb(color, [255, 102, 0])
  const draw: GraphicsDict = {
    _type: 'graphics',
    shape: 'arrow',
    pts: [[x1, y1], [x2, y2]],
    relative: true,
    color,
    r, g, b,
    thickness: Math.max(1, Math.round(Number(params.thickness) || 2)),
    tip_length: Number(params.tip_length) || 0.3,
  }
  return { draw }
}

export const drawEllipse: NodeImpl = (inputs, params) => {
  const cx = scalarOrParam(inputs, params, 'cx', 0.5)
  const cy = scalarOrParam(inputs, params, 'cy', 0.5)
  const rx = scalarOrParam(inputs, params, 'rx', 0.2)
  const ry = scalarOrParam(inputs, params, 'ry', 0.1)
  const angle = scalarOrParam(inputs, params, 'angle', 0)
  const color = String(params.color ?? '#00FF00')
  const [r, g, b] = hexToRgb(color, [0, 255, 0])
  const draw: GraphicsDict = {
    _type: 'graphics',
    shape: 'ellipse',
    pts: [[cx, cy]],
    relative: true,
    rx, ry, angle,
    color,
    r, g, b,
    thickness: Math.max(1, Math.round(Number(params.thickness) || 2)),
    fill: !!params.fill,
  }
  return { draw }
}

export const drawLine: NodeImpl = (inputs, params) => {
  const x1 = scalarOrParam(inputs, params, 'x1', 0.1)
  const y1 = scalarOrParam(inputs, params, 'y1', 0.1)
  const x2 = scalarOrParam(inputs, params, 'x2', 0.9)
  const y2 = scalarOrParam(inputs, params, 'y2', 0.9)
  const color = String(params.color ?? '#00FF00')
  const [r, g, b] = hexToRgb(color, [0, 255, 0])
  const draw: GraphicsDict = {
    _type: 'graphics',
    shape: 'line',
    pts: [[x1, y1], [x2, y2]],
    relative: true,
    color,
    r, g, b,
    thickness: Math.max(1, Math.round(Number(params.thickness) || 2)),
  }
  return { draw }
}

export const drawPoint: NodeImpl = (inputs, params) => {
  const x = scalarOrParam(inputs, params, 'x', 0.5)
  const y = scalarOrParam(inputs, params, 'y', 0.5)
  const color = String(params.color ?? '#FF0000')
  const [r, g, b] = hexToRgb(color, [255, 0, 0])
  const draw: GraphicsDict = {
    _type: 'graphics',
    shape: 'point',
    pts: [[x, y]],
    relative: true,
    color,
    r, g, b,
    thickness: Math.max(1, Math.round(Number(params.thickness) || 5)),
  }
  return { draw }
}

export const drawRect: NodeImpl = (inputs, params) => {
  const x1 = scalarOrParam(inputs, params, 'x1', 0.2)
  const y1 = scalarOrParam(inputs, params, 'y1', 0.2)
  const x2 = scalarOrParam(inputs, params, 'x2', 0.8)
  const y2 = scalarOrParam(inputs, params, 'y2', 0.8)
  const color = String(params.color ?? '#0000FF')
  const [r, g, b] = hexToRgb(color, [0, 0, 255])
  const draw: GraphicsDict = {
    _type: 'graphics',
    shape: 'rect',
    pts: [[x1, y1], [x2, y2]],
    relative: true,
    color,
    r, g, b,
    thickness: Math.max(1, Math.round(Number(params.thickness) || 2)),
    fill: !!params.fill,
  }
  return { draw }
}

export const drawText: NodeImpl = (inputs, params, ctx) => {
  const inputText = inputs.text
  const template = typeof inputText === 'string' && inputText.trim() !== '' ? inputText : String(params.text ?? 'Hello')

  // Formats {a}..{z} placeholders from single-letter ports, like the desktop node.
  let text = template
  if (template.includes('{') && template.includes('}')) {
    try {
      text = template.replace(/\{([a-z])\}/g, (_, letter: string) => {
        const val = inputs[letter]
        if (typeof val === 'number') return Number.isInteger(val) ? String(val) : val.toFixed(2)
        if (val !== undefined && val !== null) return String(val)
        return '---'
      })
    } catch {
      text = template
    }
  }

  const x = scalarOrParam(inputs, params, 'x', 0.5)
  const y = scalarOrParam(inputs, params, 'y', 0.5)
  const scale = Number(params.font_scale) || 1.0
  const thickness = Math.max(1, Math.round(Number(params.thickness) || 2))
  const color = String(params.color ?? '#FFFFFF')
  const [r, g, b] = hexToRgb(color, [255, 255, 255])

  const graphic: GraphicsDict = {
    _type: 'graphics',
    shape: 'text',
    text,
    pts: [[x, y]],
    relative: true,
    font_scale: scale,
    thickness,
    color,
    r, g, b,
  }

  const image = inputs.image as any
  if (!image) return { main: null, graphic }

  const cv = ctx.cv
  const overlay = ctx.track(toBgr(cv, image))
  const px = Math.round(x * overlay.cols)
  const py = Math.round(y * overlay.rows)
  cv.putText(overlay, text, new cv.Point(px, py), cv.FONT_HERSHEY_SIMPLEX, scale, new cv.Scalar(b, g, r, 255), thickness, cv.LINE_AA)
  return { main: overlay, graphic }
}

// --- compositing -------------------------------------------------------------

/** `fallbackRgb` is [r, g, b], matching parseColor's own convention. */
function resolveColor(cv: any, data: Record<string, unknown>, fallbackRgb: [number, number, number]): any {
  if (typeof data.color === 'string') return parseColor(cv, data.color, fallbackRgb)
  if (typeof data.r === 'number' && typeof data.g === 'number' && typeof data.b === 'number') {
    return new cv.Scalar(data.b, data.g, data.r, 255)
  }
  return new cv.Scalar(fallbackRgb[2], fallbackRgb[1], fallbackRgb[0], 255)
}

function drawGraphics(cv: any, img: any, data: GraphicsDict, w: number, h: number): void {
  const shape = data.shape ?? 'point'
  const pts = data.pts ?? []
  const relative = data.relative !== false
  const color = resolveColor(cv, data as any, [0, 255, 0])
  const thickness = Math.max(1, Math.round(Number(data.thickness) || 2))
  const fill = !!data.fill

  const scaled = pts.map(([px, py]) =>
    relative ? new cv.Point(Math.round(px * w), Math.round(py * h)) : new cv.Point(Math.round(px), Math.round(py))
  )
  if (scaled.length === 0 && shape !== 'text') return

  if (shape === 'point' && scaled.length > 0) {
    cv.circle(img, scaled[0], Math.max(1, thickness), color, -1)
  } else if (shape === 'line' && scaled.length >= 2) {
    cv.line(img, scaled[0], scaled[1], color, thickness, cv.LINE_AA)
  } else if (shape === 'rect' && scaled.length >= 2) {
    cv.rectangle(img, scaled[0], scaled[1], color, fill ? -1 : thickness)
    if (data.label) cv.putText(img, data.label, new cv.Point(scaled[0].x, scaled[0].y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
  } else if (shape === 'polygon' && scaled.length > 2) {
    const points = scaled.map((p: any) => ({ x: p.x, y: p.y }))
    if (fill) {
      const flat = points.flatMap((p) => [p.x, p.y])
      const pointsMat = cv.matFromArray(points.length, 1, cv.CV_32SC2, flat)
      const vector = new cv.MatVector()
      vector.push_back(pointsMat)
      cv.fillPoly(img, vector, color)
      pointsMat.delete()
      vector.delete()
    }
    drawPolyline(cv, img, points, true, color, thickness)
    if (data.label) cv.putText(img, data.label, new cv.Point(scaled[0].x, scaled[0].y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
  } else if (shape === 'circle' && scaled.length > 0) {
    const radius = relative ? Math.round((data.radius_rel ?? data.radius ?? 0.1) * w) : Math.round(data.radius ?? 10)
    cv.circle(img, scaled[0], radius, color, thickness)
    if (data.label) cv.putText(img, data.label, new cv.Point(scaled[0].x - radius, scaled[0].y - radius - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
  } else if (shape === 'ellipse' && scaled.length > 0) {
    const rx = relative ? Math.round((data.rx ?? 0.2) * w) : Math.round(data.rx ?? 20)
    const ry = relative ? Math.round((data.ry ?? 0.1) * h) : Math.round(data.ry ?? 10)
    const angle = Number(data.angle) || 0
    cv.ellipse(img, scaled[0], new cv.Size(rx, ry), angle, 0, 360, color, fill ? -1 : thickness, cv.LINE_AA)
    if (data.label) cv.ellipse(img, scaled[0], new cv.Size(rx, ry), angle, 0, 360, color, thickness, cv.LINE_AA)
  } else if (shape === 'arrow' && scaled.length >= 2) {
    drawArrowedLine(cv, img, scaled[0], scaled[1], color, thickness, Number(data.tip_length) || 0.3)
  } else if (shape === 'text' && scaled.length > 0) {
    const text = String(data.text ?? data.label ?? '')
    const scale = Number(data.font_scale) || 1.0
    cv.putText(img, text, scaled[0], cv.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv.LINE_AA)
  }
}

export const drawOverlay: NodeImpl = (inputs, _params, ctx) => {
  const img = (inputs.image ?? inputs.main ?? inputs.raw_frame) as any
  if (!img) return { main: null }
  const cv = ctx.cv
  const overlay = ctx.track(toBgr(cv, img))
  const w = overlay.cols
  const h = overlay.rows
  const defaultColor = new cv.Scalar(0, 255, 0, 255)

  for (const [key, data] of Object.entries(inputs)) {
    if (data === null || data === undefined || key === 'image' || key === 'main' || key === 'raw_frame') continue
    const items = Array.isArray(data) ? data : [data]

    for (let idx = 0; idx < items.length; idx++) {
      const item = items[idx]
      if (item === null || item === undefined) continue

      if (typeof item === 'object' && !Array.isArray(item) && (item as any)._type === 'graphics') {
        drawGraphics(cv, overlay, item as GraphicsDict, w, h)
        continue
      }

      if (Array.isArray(item) && item.length > 0) {
        const pts = item
          .filter((p) => Array.isArray(p) && p.length >= 2)
          .map((p: number[]) => new cv.Point(Math.round(p[0] * w), Math.round(p[1] * h)))
        if (pts.length > 2) {
          const color = new cv.Scalar(
            ((idx * 197 + 120) % 200) + 55,
            ((idx * 137 + 80) % 200) + 55,
            ((idx * 67 + 40) % 200) + 55,
            255
          )
          drawPolyline(cv, overlay, pts.map((p: any) => ({ x: p.x, y: p.y })), true, color, 2)
        } else {
          for (const p of pts) cv.circle(overlay, p, 2, defaultColor, -1)
        }
        continue
      }

      if (typeof item === 'object' && !Array.isArray(item)) {
        const dict = item as Record<string, unknown>
        if (typeof dict.xmin === 'number' && typeof dict.ymin === 'number') {
          const x1 = Math.round(dict.xmin * w)
          const y1 = Math.round(dict.ymin * h)
          let x2: number, y2: number
          if (typeof dict.width === 'number' && typeof dict.height === 'number') {
            x2 = x1 + Math.round(dict.width * w)
            y2 = y1 + Math.round(dict.height * h)
          } else if (typeof dict.xmax === 'number' && typeof dict.ymax === 'number') {
            x2 = Math.round(dict.xmax * w)
            y2 = Math.round(dict.ymax * h)
          } else {
            continue
          }
          const color = resolveColor(cv, dict, [0, 255, 0])
          const thickness = Math.max(1, Math.round(Number(dict.thickness) || 2))
          cv.rectangle(overlay, new cv.Point(x1, y1), new cv.Point(x2, y2), color, thickness)
          let label = String(dict.label ?? dict.class ?? '')
          const conf = dict.confidence ?? dict.score
          if (typeof conf === 'number') label += ` ${conf.toFixed(2)}`
          if (label) cv.putText(overlay, label, new cv.Point(x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)
          continue
        }

        if (Array.isArray(dict.landmarks)) {
          const color = resolveColor(cv, dict, [0, 0, 255])
          for (const lm of dict.landmarks as { x: number; y: number }[]) {
            cv.circle(overlay, new cv.Point(Math.round(lm.x * w), Math.round(lm.y * h)), 2, color, -1)
          }
        }
      }
    }
  }

  return { main: overlay }
}

export const drawTintMask: NodeImpl = (inputs, params, ctx) => {
  const image = inputs.image as any
  const mask = inputs.mask as any
  if (!image) return {}
  const cv = ctx.cv
  const color = parseColor(cv, String(params.color ?? '#00ff88'))
  const alpha = Math.max(0, Math.min(1, Number(params.alpha) ?? 0.5))

  const vis = ctx.track(toBgr(cv, image))
  if (!mask) return { main: vis }

  const gray = ctx.track(new cv.Mat())
  if (mask.channels() === 1) mask.copyTo(gray)
  else cv.cvtColor(mask, gray, cv.COLOR_BGR2GRAY)
  const binary = ctx.track(new cv.Mat())
  cv.threshold(gray, binary, 127, 255, cv.THRESH_BINARY)

  const tinted = ctx.track(vis.clone())
  const maskData = binary.data as Uint8Array
  const visData = vis.data as Uint8Array
  const tintedData = tinted.data as Uint8Array
  const [b, g, r] = [color[0], color[1], color[2]]
  const channels = vis.channels()
  for (let i = 0, px = 0; i < maskData.length; i++, px += channels) {
    if (maskData[i] === 0) continue
    tintedData[px] = Math.round(visData[px] * (1 - alpha) + b * alpha)
    tintedData[px + 1] = Math.round(visData[px + 1] * (1 - alpha) + g * alpha)
    tintedData[px + 2] = Math.round(visData[px + 2] * (1 - alpha) + r * alpha)
  }

  return { main: tinted }
}
