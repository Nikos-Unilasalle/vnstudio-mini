import type { NodeImpl } from '../types'
import { colorizeLabels, computeLabelStats, huMoments, toBgr, toGray } from '../cvUtils'

export const sciMarkerFilter: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.markers as any
  if (!src) return { markers: null, count: 0 }
  const cv = ctx.cv
  const minArea = Number(params.min_area) ?? 200
  const maxArea = Number(params.max_area) ?? 1000000

  const dst = ctx.track(new cv.Mat())
  src.copyTo(dst)
  const stats = computeLabelStats(src)
  const data = dst.data32S as Int32Array
  for (let i = 0; i < data.length; i++) {
    const label = data[i]
    if (label <= 0) continue
    const s = stats.get(label)
    if (!s || s.area < minArea || s.area > maxArea) data[i] = 0
  }

  return { markers: dst, count: computeLabelStats(dst).size }
}

export interface MeasuredRegion {
  id: number
  area: number
  equivalent_diameter: number
  centroid_x: number
  centroid_y: number
  bbox_width: number
  bbox_height: number
  mean_intensity: number | null
  /** Present only when a calibration is connected — undefined means "pixels only". */
  equivalent_diameter_um?: number
  area_um2?: number
}

export const sciRegionProps: NodeImpl = (inputs, params, ctx) => {
  const labels = inputs.labels_map as any
  if (!labels) return { regions: [], count: 0, main: null }
  const cv = ctx.cv

  const connected = typeof inputs.um_per_px === 'number' ? (inputs.um_per_px as number) : null
  const umPerPx = connected ?? (Number(params.um_per_px) || 0)
  const calibrated = umPerPx > 0 && umPerPx !== 1

  const intensityImage = inputs.image as any
  let gray: any = null
  if (intensityImage && params.intensity !== false) gray = ctx.track(toGray(cv, intensityImage))

  const stats = computeLabelStats(labels)
  const labelData = labels.data32S as Int32Array
  const intensitySums = new Map<number, number>()
  if (gray) {
    const grayData = gray.data as Uint8Array
    for (let i = 0; i < labelData.length; i++) {
      const label = labelData[i]
      if (label > 0) intensitySums.set(label, (intensitySums.get(label) ?? 0) + grayData[i])
    }
  }

  const regions: MeasuredRegion[] = []
  for (const [id, s] of stats) {
    const equivalentDiameter = 2 * Math.sqrt(s.area / Math.PI)
    const region: MeasuredRegion = {
      id,
      area: s.area,
      equivalent_diameter: equivalentDiameter,
      centroid_x: s.cx,
      centroid_y: s.cy,
      bbox_width: s.maxX - s.minX + 1,
      bbox_height: s.maxY - s.minY + 1,
      mean_intensity: gray ? (intensitySums.get(id) ?? 0) / s.area : null,
    }
    if (calibrated) {
      region.equivalent_diameter_um = equivalentDiameter * umPerPx
      region.area_um2 = s.area * umPerPx * umPerPx
    }
    regions.push(region)
  }

  const preview = ctx.track(colorizeLabels(cv, labels))
  if (intensityImage) {
    const base = ctx.track(toBgr(cv, intensityImage))
    if (base.rows === preview.rows && base.cols === preview.cols) {
      cv.addWeighted(base, 0.5, preview, 0.5, 0, preview)
    }
  }

  if (params.show_ids) {
    for (const region of regions) {
      cv.putText(
        preview,
        String(region.id),
        new cv.Point(Math.round(region.centroid_x) - 8, Math.round(region.centroid_y) + 4),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        new cv.Scalar(255, 255, 255, 255),
        1,
        cv.LINE_AA
      )
    }
  }

  ctx.emit('count', regions.length)
  return { regions, count: regions.length, main: preview }
}

const MICRONS_PER_UNIT: Record<string, number> = { 'µm': 1, um: 1, mm: 1000, cm: 10000, m: 1000000, in: 25400 }

export const sciInteractiveCalibration: NodeImpl = (inputs, params, ctx) => {
  const image = inputs.image as any
  if (!image) return { factor: 0, um_per_px: 0, unit: String(params.unit ?? 'mm'), main: null }

  let points: { x: number; y: number }[] = []
  try {
    const parsed = JSON.parse(String(params.points ?? '[]'))
    if (Array.isArray(parsed)) points = parsed
  } catch {
    points = []
  }

  const unit = String(params.unit ?? 'mm')
  if (points.length !== 2) {
    ctx.emit('display_value', 'trace une ligne')
    return { factor: 0, um_per_px: 0, unit, main: image }
  }

  const cv = ctx.cv
  const width = image.cols
  const height = image.rows
  const a = { x: points[0].x * width, y: points[0].y * height }
  const b = { x: points[1].x * width, y: points[1].y * height }
  const pixelLength = Math.hypot(b.x - a.x, b.y - a.y)
  const realLength = Number(params.real_len) || 10

  const pxPerUnit = pixelLength > 0 ? pixelLength / realLength : 0
  const micronsPerUnit = MICRONS_PER_UNIT[unit] ?? 0
  const umPerPx = micronsPerUnit && pixelLength > 0 ? (realLength * micronsPerUnit) / pixelLength : 0

  const overlay = ctx.track(toBgr(cv, image))
  const thickness = Math.max(1, Math.round(Math.max(width, height) / 400))
  cv.line(overlay, new cv.Point(Math.round(a.x), Math.round(a.y)), new cv.Point(Math.round(b.x), Math.round(b.y)), new cv.Scalar(255, 0, 255, 255), thickness, cv.LINE_AA)
  cv.circle(overlay, new cv.Point(Math.round(a.x), Math.round(a.y)), thickness * 3, new cv.Scalar(255, 0, 255, 255), -1)
  cv.circle(overlay, new cv.Point(Math.round(b.x), Math.round(b.y)), thickness * 3, new cv.Scalar(255, 0, 255, 255), -1)

  ctx.emit('display_value', `${pxPerUnit.toFixed(2)} px/${unit} · ${umPerPx.toFixed(1)} µm/px`)
  return { factor: pxPerUnit, um_per_px: umPerPx, unit, main: overlay }
}

export const sciCalibration: NodeImpl = (inputs, params, ctx) => {
  const value = inputs.input
  if (value === undefined || value === null) return { output: null }

  const factor = Number(params.factor) || 100
  const isArea = String(params.dimension ?? 'Area') === 'Area'
  const divisor = factor <= 0 ? 1 : isArea ? factor * factor : factor
  const unit = `${params.unit_name ?? 'cm'}${isArea ? '²' : ''}`

  if (Array.isArray(value)) {
    const converted = value.map((v) => (typeof v === 'number' ? v / divisor : v))
    ctx.emit('display_value', `${converted.length} items`)
    return { output: converted }
  }

  const numeric = Number(value)
  if (Number.isNaN(numeric)) return { output: value }
  const converted = numeric / divisor
  ctx.emit('display_value', `${converted.toFixed(3)} ${unit}`)
  return { output: converted }
}

export const imageMoments: NodeImpl = (inputs, params, ctx) => {
  const image = inputs.image as any
  if (!image) return { main: null, data: null }
  const cv = ctx.cv

  const binary = ctx.track(new cv.Mat())
  const maskIn = inputs.mask as any
  if (maskIn) {
    const gray = ctx.track(toGray(cv, maskIn))
    cv.threshold(gray, binary, 127, 255, cv.THRESH_BINARY)
  } else {
    const gray = ctx.track(toGray(cv, image))
    cv.threshold(gray, binary, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
  }

  let moments: any
  let contour: any = null
  if (String(params.source ?? 'Largest Contour') === 'Largest Contour') {
    const contours = new cv.MatVector()
    const hierarchy = ctx.track(new cv.Mat())
    cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if (contours.size() > 0) {
      let best = contours.get(0)
      let bestArea = cv.contourArea(best)
      for (let i = 1; i < contours.size(); i++) {
        const candidate = contours.get(i)
        const area = cv.contourArea(candidate)
        if (area > bestArea) {
          bestArea = area
          best = candidate
        }
      }
      // Clone before releasing the vector: `best` points into memory the vector owns.
      contour = ctx.track(best.clone())
      moments = cv.moments(best, false)
    } else {
      moments = cv.moments(binary, true)
    }
    contours.delete()
  } else {
    moments = cv.moments(binary, true)
  }

  const m00 = moments.m00
  const cx = m00 !== 0 ? moments.m10 / m00 : 0
  const cy = m00 !== 0 ? moments.m01 / m00 : 0
  const { mu20, mu02, mu11, mu30, mu03 } = moments

  const theta = 0.5 * ((Math.atan2(2 * mu11, mu20 - mu02) * 180) / Math.PI)
  const spread = mu20 + mu02
  const anisotropy = spread > 0 ? Math.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2) / spread : 0

  let semiMajor = 0
  let semiMinor = 0
  let eccentricity = 0
  if (spread > 0) {
    const term = Math.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2)
    const lambda1 = (spread + term) / 2
    const lambda2 = (spread - term) / 2
    semiMajor = 2 * Math.sqrt(Math.max(lambda1, 0))
    semiMinor = 2 * Math.sqrt(Math.max(lambda2, 0))
    eccentricity = lambda1 > 0 && lambda2 >= 0 ? Math.sqrt(1 - lambda2 / lambda1) : 0
  }

  // Hu values span many orders of magnitude; the desktop node log-scales them
  // for readability, preserving sign.
  const huLog = huMoments(moments).map((v) => (v === 0 ? 0 : -Math.sign(v) * Math.log10(Math.abs(v))))

  const overlay = ctx.track(toBgr(cv, image))
  if (params.draw_overlay !== false && m00 !== 0) {
    const x = Math.round(cx)
    const y = Math.round(cy)
    if (contour) {
      const single = new cv.MatVector()
      single.push_back(contour)
      cv.drawContours(overlay, single, -1, new cv.Scalar(0, 255, 255, 255), 2)
      single.delete()
    }
    cv.circle(overlay, new cv.Point(x, y), 6, new cv.Scalar(0, 0, 255, 255), -1)
    cv.line(overlay, new cv.Point(x - 14, y), new cv.Point(x + 14, y), new cv.Scalar(0, 0, 255, 255), 2)
    cv.line(overlay, new cv.Point(x, y - 14), new cv.Point(x, y + 14), new cv.Scalar(0, 0, 255, 255), 2)
  }

  if (params.draw_ellipse && m00 !== 0 && semiMajor > 1) {
    const x = Math.round(cx)
    const y = Math.round(cy)
    const radians = (theta * Math.PI) / 180
    cv.ellipse(
      overlay,
      new cv.Point(x, y),
      new cv.Size(Math.max(1, Math.round(semiMajor)), Math.max(1, Math.round(semiMinor))),
      -theta,
      0,
      360,
      new cv.Scalar(0, 180, 255, 255),
      1,
      cv.LINE_AA
    )
    const dx = Math.round(semiMajor * Math.cos(radians))
    const dy = Math.round(semiMajor * Math.sin(radians))
    cv.line(overlay, new cv.Point(x - dx, y - dy), new cv.Point(x + dx, y + dy), new cv.Scalar(0, 180, 255, 255), 2, cv.LINE_AA)
  }

  const data = {
    M00: round(m00, 2),
    centroid_x: round(cx, 2),
    centroid_y: round(cy, 2),
    area: round(m00, 2),
    mu20: round(mu20, 4),
    mu02: round(mu02, 4),
    mu11: round(mu11, 4),
    mu30: round(mu30, 4),
    mu03: round(mu03, 4),
    theta_deg: round(theta, 3),
    anisotropy: round(anisotropy, 4),
    semi_major: round(semiMajor, 2),
    semi_minor: round(semiMinor, 2),
    eccentricity: round(eccentricity, 4),
    ...Object.fromEntries(huLog.map((v, i) => [`phi${i + 1}`, round(v, 4)])),
  }
  ctx.emit('report', data)
  return { main: overlay, data }
}

function round(value: number, digits: number): number {
  const factor = 10 ** digits
  return Math.round(value * factor) / factor
}

export const sciAnalysisReport: NodeImpl = (inputs, _params, ctx) => {
  const data = (inputs.data as Record<string, unknown>) ?? {}
  ctx.emit('report', data)
  return { report: data }
}
