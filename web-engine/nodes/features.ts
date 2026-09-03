import type { NodeImpl } from '../types'
import { drawPolyline, toGray } from '../cvUtils'

function hexToBgr(hex: string, fallback: [number, number, number] = [0, 255, 0]): [number, number, number] {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(hex ?? ''))
  if (!m) return fallback
  const int = parseInt(m[1], 16)
  return [int & 255, (int >> 8) & 255, (int >> 16) & 255]
}

function hueToHex(hue: number): string {
  // Matches cv2.cvtColor(HSV[hue,220,230] -> BGR): OpenCV hue range is [0,180).
  const h = (hue % 180) / 180
  const s = 220 / 255
  const v = 230 / 255
  const i = Math.floor(h * 6)
  const f = h * 6 - i
  const p = v * (1 - s)
  const q = v * (1 - f * s)
  const t = v * (1 - (1 - f) * s)
  let r = 0, g = 0, b = 0
  switch (i % 6) {
    case 0: r = v; g = t; b = p; break
    case 1: r = q; g = v; b = p; break
    case 2: r = p; g = v; b = t; break
    case 3: r = p; g = q; b = v; break
    case 4: r = t; g = p; b = v; break
    case 5: r = v; g = p; b = q; break
  }
  const toHex = (x: number) => Math.round(x * 255).toString(16).padStart(2, '0')
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`
}

interface ContourItem {
  id: number
  label: string
  _type: 'graphics'
  shape: 'polygon'
  pts: [number, number][]
  area: number
  elongation: number
  circularity: number
  angle: number
  center: { x: number; y: number }
  relative: true
  color: string
}

const CONTOUR_MODES = ['RETR_EXTERNAL', 'RETR_LIST', 'RETR_CCOMP', 'RETR_TREE'] as const
const CONTOUR_METHODS = ['CHAIN_APPROX_NONE', 'CHAIN_APPROX_SIMPLE', 'CHAIN_APPROX_TC89_L1', 'CHAIN_APPROX_TC89_KCOS'] as const

export const featFindContours: NodeImpl = (inputs, params, ctx) => {
  const mask = inputs.mask as any
  if (!mask) return { contours_list: [], count: 0 }
  const cv = ctx.cv
  const gray = ctx.track(toGray(cv, mask))

  const modeKey = CONTOUR_MODES[Number(params.mode) || 0] ?? 'RETR_EXTERNAL'
  const methodKey = CONTOUR_METHODS[Number(params.method) ?? 1] ?? 'CHAIN_APPROX_SIMPLE'
  const minArea = Number(params.min_area) ?? 100
  const maxArea = Number(params.max_area) || 0
  const epsilon = Number(params.epsilon) || 0

  const contours = new cv.MatVector()
  const hierarchy = ctx.track(new cv.Mat())
  cv.findContours(gray, contours, hierarchy, cv[modeKey], cv[methodKey])

  const w = gray.cols
  const h = gray.rows
  const results: ContourItem[] = []
  let rank = 0

  for (let i = 0; i < contours.size(); i++) {
    let cnt = contours.get(i)
    let simplified: any = null
    if (epsilon > 0) {
      simplified = new cv.Mat()
      cv.approxPolyDP(cnt, simplified, epsilon, true)
      cnt = simplified
    }

    const area = cv.contourArea(cnt)
    if (area < minArea || (maxArea > 0 && area > maxArea)) {
      if (simplified) simplified.delete()
      continue
    }

    const ptsData = cnt.data32S as Int32Array
    const pts: [number, number][] = []
    for (let p = 0; p < ptsData.length; p += 2) pts.push([ptsData[p] / w, ptsData[p + 1] / h])

    const moments = cv.moments(cnt)
    const cx = moments.m00 !== 0 ? moments.m10 / moments.m00 : 0
    const cy = moments.m00 !== 0 ? moments.m01 / moments.m00 : 0

    const rect = cv.minAreaRect(cnt)
    const rw = rect.size.width
    const rh = rect.size.height
    const elongation = Math.min(rw, rh) > 0 ? Math.max(rw, rh) / Math.min(rw, rh) : 1
    const angle = rect.angle

    const perimeter = cv.arcLength(cnt, true)
    const circularity = perimeter > 0 ? (4 * Math.PI * area) / (perimeter * perimeter) : 1

    const color = hueToHex((rank * 47) % 180)

    results.push({
      id: rank,
      label: `#${rank}`,
      _type: 'graphics',
      shape: 'polygon',
      pts,
      area,
      elongation: Number(elongation.toFixed(3)),
      circularity: Number(circularity.toFixed(4)),
      angle: Number(angle.toFixed(2)),
      center: { x: cx / w, y: cy / h },
      relative: true,
      color,
    })
    rank++
    if (simplified) simplified.delete()
  }

  contours.delete()
  return { contours_list: results, count: results.length }
}

export const featContourProps: NodeImpl = (inputs) => {
  const c = inputs.contour as Record<string, any> | undefined
  if (!c || typeof c !== 'object') return { area: 0, circularity: 0, elongation: 1, center_x: 0, center_y: 0 }
  return {
    area: c.area ?? 0,
    circularity: c.circularity ?? 0,
    elongation: c.elongation ?? 1,
    center_x: c.center?.x ?? 0,
    center_y: c.center?.y ?? 0,
  }
}

export const featBilateral: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv
  const d = Math.round(Number(params.diameter) || 9)
  const sigmaColor = Number(params.sigma_color) || 75
  const sigmaSpace = Number(params.sigma_space) || 75
  const dst = ctx.track(new cv.Mat())
  cv.bilateralFilter(src, dst, d, sigmaColor, sigmaSpace, cv.BORDER_DEFAULT)
  return { main: dst }
}

export const featHoughCircles: NodeImpl = (inputs, params, ctx) => {
  const image = inputs.image as any
  const maskIn = inputs.mask as any
  const source = image ?? maskIn
  if (!source) return { main: null, mask: null, labels_map: null, circles_list: [], count: 0 }
  const cv = ctx.cv

  const w = source.cols
  const h = source.rows
  let gray = ctx.track(toGray(cv, source))

  if (image && maskIn) {
    let m = maskIn
    if (m.cols !== w || m.rows !== h) {
      const resized = ctx.track(new cv.Mat())
      cv.resize(m, resized, new cv.Size(w, h), 0, 0, cv.INTER_NEAREST)
      m = resized
    }
    const mGray = ctx.track(toGray(cv, m))
    const masked = ctx.track(new cv.Mat())
    cv.bitwise_and(gray, gray, masked, mGray)
    gray = masked
  }

  const dp = Number(params.dp) || 1.2
  const minDist = Number(params.min_dist) || 100
  const p1 = Number(params.param1) || 100
  const p2 = Number(params.param2) || 30
  const minR = Math.round(Number(params.min_r) || 0)
  const maxR = Math.round(Number(params.max_r) || 0)
  const [bB, bG, bR] = hexToBgr(String(params.viz_color ?? '#00FF00'))
  const thickness = Math.round(Number(params.thickness) ?? 2)

  const circlesMat = ctx.track(new cv.Mat())
  cv.HoughCircles(gray, circlesMat, cv.HOUGH_GRADIENT, dp, minDist, p1, p2, minR, maxR)

  const outImg = ctx.track(image ? (() => { const o = new cv.Mat(); toBgrTrack(cv, source, o); return o })() : new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(0, 0, 0)))
  const mask = ctx.track(new cv.Mat(h, w, cv.CV_8U, new cv.Scalar(0)))
  const labelsMap = ctx.track(new cv.Mat(h, w, cv.CV_32S, new cv.Scalar(0)))

  const results: any[] = []
  const data = circlesMat.data32F as Float32Array
  const color = new cv.Scalar(bB, bG, bR, 255)
  for (let i = 0; i < data.length / 3; i++) {
    const cx = Math.round(data[i * 3])
    const cy = Math.round(data[i * 3 + 1])
    const r = Math.round(data[i * 3 + 2])
    cv.circle(outImg, new cv.Point(cx, cy), r, color, thickness, cv.LINE_AA)
    cv.circle(mask, new cv.Point(cx, cy), r, new cv.Scalar(255), -1)
    cv.circle(labelsMap, new cv.Point(cx, cy), r, new cv.Scalar(i + 1), -1)
    results.push({
      id: i + 1,
      label: `circle_${i + 1}`,
      _type: 'graphics',
      shape: 'circle',
      pts: [[cx / w, cy / h]],
      radius: r,
      radius_rel: r / w,
      area: Math.PI * r * r,
      relative: true,
      color: `#${bR.toString(16).padStart(2, '0')}${bG.toString(16).padStart(2, '0')}${bB.toString(16).padStart(2, '0')}`,
    })
  }

  return { main: outImg, mask, labels_map: labelsMap, circles_list: results, count: results.length }
}

function toBgrTrack(cv: any, src: any, out: any): void {
  if (src.channels() === 1) cv.cvtColor(src, out, cv.COLOR_GRAY2BGR)
  else if (src.channels() === 4) cv.cvtColor(src, out, cv.COLOR_BGRA2BGR)
  else src.copyTo(out)
}

export const featFilterContours: NodeImpl = (inputs, params, ctx) => {
  const contours = (inputs.contours as Record<string, any>[]) ?? []
  const image = inputs.image as any

  const maxCirc = Number(params.max_circularity) || 0
  const minCirc = Number(params.min_circularity) || 0
  const minElo = Number(params.min_elongation) ?? 1
  const maxElo = Number(params.max_elongation) || 0
  const minArea = Number(params.min_area) || 0
  const maxArea = Number(params.max_area) || 0

  const keep = (c: Record<string, any>) => {
    const circ = Number(c.circularity ?? 1)
    const elo = Number(c.elongation ?? 1)
    const area = Number(c.area ?? 0)
    if (maxCirc > 0 && circ > maxCirc) return false
    if (minCirc > 0 && circ < minCirc) return false
    if (minElo > 1 && elo < minElo) return false
    if (maxElo > 0 && elo > maxElo) return false
    if (minArea > 0 && area < minArea) return false
    if (maxArea > 0 && area > maxArea) return false
    return true
  }

  const results: Record<string, any>[] = []
  const rejected: Record<string, any>[] = []
  for (const c of contours) {
    if (!c || typeof c !== 'object') continue
    ;(keep(c) ? results : rejected).push(c)
  }

  let overlay: any = null
  if (image) {
    const cv = ctx.cv
    overlay = ctx.track((() => { const o = new cv.Mat(); toBgrTrack(cv, image, o); return o })())
    const w = overlay.cols
    const h = overlay.rows
    const thickness = Math.max(1, Math.round(Number(params.thickness) || 2))
    const doFill = !!params.fill

    const toPx = (c: Record<string, any>) => {
      const raw = c.pts as [number, number][] | undefined
      if (!raw || raw.length < 3) return null
      const rel = c.relative !== false
      return raw.map(([x, y]) => (rel ? { x: x * w, y: y * h } : { x, y }))
    }

    if (params.show_rejected !== false) {
      for (const c of rejected) {
        const px = toPx(c)
        if (px) drawPolyline(cv, overlay, px, true, new cv.Scalar(60, 60, 200, 255), 1)
      }
    }

    for (const c of results) {
      const px = toPx(c)
      if (!px) continue
      const [b, g, r] = hexToBgr(String(c.color ?? '#00ff00'))
      const color = new cv.Scalar(b, g, r, 255)
      if (doFill) {
        const flat = px.flatMap((p) => [Math.round(p.x), Math.round(p.y)])
        const pointsMat = cv.matFromArray(px.length, 1, cv.CV_32SC2, flat)
        const vector = new cv.MatVector()
        vector.push_back(pointsMat)
        const layer = ctx.track(overlay.clone())
        cv.fillPoly(layer, vector, color)
        const blended = ctx.track(new cv.Mat())
        cv.addWeighted(overlay, 0.6, layer, 0.4, 0, blended)
        blended.copyTo(overlay)
        pointsMat.delete()
        vector.delete()
      }
      drawPolyline(cv, overlay, px, true, color, thickness)
    }
  }

  return { contours_list: results, main: overlay, count: results.length }
}

export const featFillContours: NodeImpl = (inputs, params, ctx) => {
  const contours = (inputs.contours as Record<string, any>[]) ?? []
  const image = inputs.image as any
  const cv = ctx.cv

  let w: number, h: number, out: any
  if (image) {
    w = image.cols
    h = image.rows
    out = ctx.track((() => { const o = new cv.Mat(); toBgrTrack(cv, image, o); return o })())
  } else {
    w = Math.round(Number(params.width) || 512)
    h = Math.round(Number(params.height) || 512)
    out = ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(0, 0, 0)))
  }

  const mask = ctx.track(new cv.Mat(h, w, cv.CV_8U, new cv.Scalar(0)))

  for (const c of contours) {
    if (!c || typeof c !== 'object' || !c.pts) continue
    const rel = c.relative !== false
    const raw = c.pts as [number, number][]
    if (raw.length < 3) continue
    const flat = rel
      ? raw.flatMap(([x, y]) => [Math.round(x * w), Math.round(y * h)])
      : raw.flatMap(([x, y]) => [Math.round(x), Math.round(y)])
    const pointsMat = cv.matFromArray(raw.length, 1, cv.CV_32SC2, flat)
    const vector = new cv.MatVector()
    vector.push_back(pointsMat)
    cv.fillPoly(mask, vector, new cv.Scalar(255))
    const [b, g, r] = hexToBgr(String(c.color ?? '#00ff00'))
    cv.fillPoly(out, vector, new cv.Scalar(b, g, r, 255))
    pointsMat.delete()
    vector.delete()
  }

  return { mask, main: out }
}

export const featHoughLines: NodeImpl = (inputs, params, ctx) => {
  const image = inputs.image as any
  if (!image) return { main: null, lines_list: [], count: 0 }
  const cv = ctx.cv

  const gray = ctx.track(toGray(cv, image))
  const rho = Number(params.rho) || 1
  const theta = ((Number(params.theta_deg) || 1) * Math.PI) / 180
  const threshold = Math.round(Number(params.threshold) || 50)
  const minLen = Number(params.min_len) || 50
  const maxGap = Number(params.max_gap) || 10
  const thickness = Math.max(1, Math.round(Number(params.thickness) || 2))

  const lines = ctx.track(new cv.Mat())
  cv.HoughLinesP(gray, lines, rho, theta, threshold, minLen, maxGap)

  const overlay = ctx.track((() => { const o = new cv.Mat(); toBgrTrack(cv, image, o); return o })())
  const w = gray.cols
  const h = gray.rows
  const results: any[] = []
  const data = lines.data32S as Int32Array
  const green = new cv.Scalar(0, 255, 0, 255)
  for (let i = 0; i < data.length / 4; i++) {
    const x1 = data[i * 4]
    const y1 = data[i * 4 + 1]
    const x2 = data[i * 4 + 2]
    const y2 = data[i * 4 + 3]
    cv.line(overlay, new cv.Point(x1, y1), new cv.Point(x2, y2), green, thickness, cv.LINE_AA)
    results.push({
      id: i,
      _type: 'graphics',
      shape: 'line',
      pts: [[x1 / w, y1 / h], [x2 / w, y2 / h]],
      relative: true,
      color: '#00ff00',
    })
  }

  return { main: overlay, lines_list: results, count: results.length }
}
