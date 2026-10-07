import type { NodeImpl } from '../types'
import type { LabelStat } from '../cvUtils'
import { colorizeLabels, computeLabelStats, huMoments, inputLabels32S, toBgr, toGray } from '../cvUtils'

export const sciMarkerFilter: NodeImpl = (inputs, params, ctx) => {
  const srcIn = inputs.markers as any
  if (!srcIn) return { markers: null, count: 0 }
  const cv = ctx.cv
  const src = inputLabels32S(ctx, srcIn)
  const minArea = Number(params.min_area ?? 200)
  const maxArea = Number(params.max_area ?? 1000000)

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
  // Shape descriptors, as the desktop plugin defines them (lengths in pixels).
  perimeter: number
  circularity: number
  aspect_ratio: number
  solidity: number
  convexity: number
  extent: number
  rectangularity: number
  roundness: number
  eccentricity: number
  anisotropy: number
  orientation: number
  feret_max: number
  feret_min: number
  perimeter_um?: number
  feret_max_um?: number
  feret_min_um?: number
  max_intensity?: number
  min_intensity?: number
  std_intensity?: number
}

type ShapeDescriptors = Pick<
  MeasuredRegion,
  | 'perimeter' | 'circularity' | 'aspect_ratio' | 'solidity' | 'convexity' | 'extent' | 'rectangularity'
  | 'roundness' | 'eccentricity' | 'anisotropy' | 'orientation' | 'feret_max' | 'feret_min'
>

/**
 * Contour- and moment-based descriptors for one label, following the desktop
 * plugin (skimage regionprops + its own contour measures): circularity from the
 * contour perimeter, solidity/convexity from the hull, rectangularity and
 * aspect ratio from the minimum-area rectangle, eccentricity, orientation and
 * anisotropy from the region's central moments.
 */
function shapeDescriptors(cv: any, labelData: Int32Array, width: number, s: LabelStat): ShapeDescriptors {
  const bw = s.maxX - s.minX + 1
  const bh = s.maxY - s.minY + 1
  // One pixel of padding so a region touching its bbox edge still gets a closed contour.
  const mask = new cv.Mat(bh + 2, bw + 2, cv.CV_8U, new cv.Scalar(0))
  const maskData = mask.data as Uint8Array
  for (let y = s.minY; y <= s.maxY; y++) {
    for (let x = s.minX; x <= s.maxX; x++) {
      if (labelData[y * width + x] === s.id) maskData[(y - s.minY + 1) * (bw + 2) + (x - s.minX + 1)] = 255
    }
  }

  const m = cv.moments(mask, true)
  const contours = new cv.MatVector()
  const hierarchy = new cv.Mat()
  cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
  let cnt: any = null
  let best = -1
  for (let i = 0; i < contours.size(); i++) {
    const c = contours.get(i)
    const a = cv.contourArea(c)
    if (a > best) { best = a; cnt?.delete(); cnt = c } else c.delete()
  }

  const area = s.area
  let perimeter = 0, solidity = 1, convexity = 1, rectangularity = 0, aspectRatio = 0
  let feretMax = 0, feretMin = 0
  if (cnt && cnt.rows >= 3) {
    perimeter = cv.arcLength(cnt, true)
    const hull = new cv.Mat()
    cv.convexHull(cnt, hull, false, true)
    const hullArea = cv.contourArea(hull)
    solidity = hullArea > 0 ? Math.min(1, area / hullArea) : 1
    convexity = perimeter > 0 ? cv.arcLength(hull, true) / perimeter : 1
    // Feret max is the widest caliper, which always spans two hull vertices.
    const hp = hull.data32S as Int32Array
    for (let i = 0; i < hp.length; i += 2) {
      for (let j = i + 2; j < hp.length; j += 2) {
        const d = Math.hypot(hp[i] - hp[j], hp[i + 1] - hp[j + 1])
        if (d > feretMax) feretMax = d
      }
    }
    hull.delete()
    const rect = cv.minAreaRect(cnt)
    const rw = rect.size.width
    const rh = rect.size.height
    rectangularity = rw * rh > 0 ? area / (rw * rh) : 0
    aspectRatio = Math.min(rw, rh) > 0 ? Math.max(rw, rh) / Math.min(rw, rh) : 0
    feretMin = Math.min(rw, rh)
  }
  cnt?.delete()
  contours.delete()
  hierarchy.delete()
  mask.delete()

  // Central moments, OpenCV's x/y naming: mu20 spreads along x (columns).
  const mu20 = m.mu20, mu02 = m.mu02, mu11 = m.mu11
  const spread = mu20 + mu02
  const half = Math.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
  const l1 = spread / 2 + half
  const l2 = spread / 2 - half
  return {
    perimeter,
    circularity: perimeter > 0 ? (4 * Math.PI * area) / (perimeter * perimeter) : 0,
    aspect_ratio: aspectRatio,
    solidity,
    convexity,
    extent: area / (bw * bh),
    rectangularity,
    roundness: feretMax > 0 ? (4 * area) / (Math.PI * feretMax * feretMax) : 0,
    eccentricity: l1 > 0 ? Math.sqrt(Math.max(0, 1 - l2 / l1)) : 0,
    anisotropy: spread > 0 ? (2 * half) / spread : 0,
    // skimage's convention: angle between the row axis and the major axis, in [-π/2, π/2].
    orientation: mu20 === mu02 ? (mu11 > 0 ? -Math.PI / 4 : mu11 < 0 ? Math.PI / 4 : 0) : 0.5 * Math.atan2(2 * mu11, mu02 - mu20),
    feret_max: feretMax,
    feret_min: feretMin,
  }
}

export const sciRegionProps: NodeImpl = (inputs, params, ctx) => {
  const labelsIn = inputs.labels_map as any
  if (!labelsIn) return { regions: [], count: 0, main: null }
  const cv = ctx.cv
  const labels = inputLabels32S(ctx, labelsIn)

  const connected = typeof inputs.um_per_px === 'number' ? (inputs.um_per_px as number) : null
  const umPerPx = connected ?? (Number(params.um_per_px) || 0)
  const calibrated = umPerPx > 0 && umPerPx !== 1

  const intensityImage = inputs.image as any
  let gray: any = null
  if (intensityImage && params.intensity !== false) gray = ctx.track(toGray(cv, intensityImage))

  const stats = computeLabelStats(labels)
  const labelData = labels.data32S as Int32Array
  const intensitySums = new Map<number, number>()
  const intensitySquares = new Map<number, number>()
  const intensityMin = new Map<number, number>()
  const intensityMax = new Map<number, number>()
  if (gray) {
    const grayData = gray.data as Uint8Array
    for (let i = 0; i < labelData.length; i++) {
      const label = labelData[i]
      if (label <= 0) continue
      const v = grayData[i]
      intensitySums.set(label, (intensitySums.get(label) ?? 0) + v)
      intensitySquares.set(label, (intensitySquares.get(label) ?? 0) + v * v)
      if (v < (intensityMin.get(label) ?? Infinity)) intensityMin.set(label, v)
      if (v > (intensityMax.get(label) ?? -Infinity)) intensityMax.set(label, v)
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
      ...shapeDescriptors(cv, labelData, labels.cols, s),
    }
    if (calibrated) {
      region.equivalent_diameter_um = equivalentDiameter * umPerPx
      region.area_um2 = s.area * umPerPx * umPerPx
      region.perimeter_um = region.perimeter * umPerPx
      region.feret_max_um = region.feret_max * umPerPx
      region.feret_min_um = region.feret_min * umPerPx
    }
    if (gray) {
      const mean = (intensitySums.get(id) ?? 0) / s.area
      region.max_intensity = intensityMax.get(id) ?? 0
      region.min_intensity = intensityMin.get(id) ?? 0
      region.std_intensity = Math.sqrt(Math.max(0, (intensitySquares.get(id) ?? 0) / s.area - mean * mean))
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
  if (value === undefined || value === null) return { main: null, output: null }

  const factor = Number(params.factor) || 100
  const isArea = String(params.dimension ?? 'Area') === 'Area'
  const divisor = factor <= 0 ? 1 : isArea ? factor * factor : factor
  const unit = `${params.unit_name ?? 'cm'}${isArea ? '²' : ''}`

  if (Array.isArray(value)) {
    const converted = value.map((v) => (typeof v === 'number' ? v / divisor : v))
    ctx.emit('display_value', `${converted.length} items`)
    return { main: converted, output: converted }
  }

  const numeric = Number(value)
  if (Number.isNaN(numeric)) return { main: value, output: value }
  const converted = numeric / divisor
  ctx.emit('display_value', `${converted.toFixed(3)} ${unit}`)
  // Desktop renamed the port to `main` (matching its UI); `output` kept for older graphs.
  return { main: converted, output: converted }
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
