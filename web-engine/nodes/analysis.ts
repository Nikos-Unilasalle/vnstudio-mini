import type { NodeImpl } from '../types'
import { toBgr, toGray } from '../cvUtils'

const ZONES_4: [string, [number, number, number]][] = [
  ['Toes', [80, 200, 120]],
  ['Forefoot', [80, 160, 240]],
  ['Arch', [240, 160, 60]],
  ['Heel', [240, 80, 60]],
]
const ZONES_3: [string, [number, number, number]][] = [
  ['Forefoot', [80, 200, 120]],
  ['Arch', [80, 160, 240]],
  ['Heel', [240, 80, 60]],
]

interface Extents {
  minY: number
  maxY: number
  minX: number
  maxX: number
  empty: boolean
}

function extentsOf(binary: any): Extents {
  const data = binary.data as Uint8Array
  const w = binary.cols
  const h = binary.rows
  let minY = h
  let maxY = -1
  let minX = w
  let maxX = -1
  for (let y = 0; y < h; y++) {
    const row = y * w
    for (let x = 0; x < w; x++) {
      if (data[row + x] === 0) continue
      if (y < minY) minY = y
      if (y > maxY) maxY = y
      if (x < minX) minX = x
      if (x > maxX) maxX = x
    }
  }
  return maxX < 0 ? { minY: 0, maxY: 0, minX: 0, maxX: 0, empty: true } : { minY, maxY, minX, maxX, empty: false }
}

/**
 * Every zone split below runs along Y, so the foot must stand upright with toes
 * at the top. Two invariants make that recoverable without asking the user: a
 * foot is longer than it is wide, and its widest line — the ball — sits in the
 * toe half rather than the heel half.
 */
function standUpright(cv: any, images: any[], binary: any): { images: any[]; binary: any } {
  const rotate = (code: number) => {
    const rotatedImages = images.map((img) => {
      const out = new cv.Mat()
      cv.rotate(img, out, code)
      return out
    })
    const rotatedBinary = new cv.Mat()
    cv.rotate(binary, rotatedBinary, code)
    return { images: rotatedImages, binary: rotatedBinary }
  }

  let extents = extentsOf(binary)
  if (extents.maxX - extents.minX > extents.maxY - extents.minY) {
    const rotated = rotate(cv.ROTATE_90_CLOCKWISE)
    images = rotated.images
    binary = rotated.binary
    extents = extentsOf(binary)
  }

  const w = binary.cols
  const data = binary.data as Uint8Array
  let widestRow = 0
  let widestCount = -1
  for (let y = extents.minY; y <= extents.maxY; y++) {
    const row = y * w
    let count = 0
    for (let x = 0; x < w; x++) if (data[row + x] > 0) count++
    if (count > widestCount) {
      widestCount = count
      widestRow = y - extents.minY
    }
  }

  if (widestCount > 0 && widestRow > (extents.maxY - extents.minY + 1) / 2) {
    const rotated = rotate(cv.ROTATE_180)
    images = rotated.images
    binary = rotated.binary
  }
  return { images, binary }
}

export const forensicFootprint: NodeImpl = (inputs, params, ctx) => {
  const image = inputs.image as any
  if (!image) return { main: null, report: {}, staheli: 0, asymmetry: 0 }
  const cv = ctx.cv

  const pxPerMmRaw = inputs.px_per_mm
  const pxPerMm = typeof pxPerMmRaw === 'number' && pxPerMmRaw > 0 ? pxPerMmRaw : 0
  const hasCalibration = pxPerMm > 0

  const zoneDefs = Number(params.n_zones) === 1 ? ZONES_3 : ZONES_4
  const zoneCount = zoneDefs.length

  let visual = ctx.track(toBgr(cv, image))
  let gray = ctx.track(toGray(cv, visual))

  let binary = ctx.track(new cv.Mat())
  const maskIn = inputs.mask as any
  if (maskIn) {
    const maskGray = ctx.track(toGray(cv, maskIn))
    cv.threshold(maskGray, binary, 127, 255, cv.THRESH_BINARY)
  } else {
    cv.threshold(gray, binary, 1, 255, cv.THRESH_BINARY)
  }

  if (extentsOf(binary).empty) return { main: visual, report: {}, staheli: 0, asymmetry: 0 }

  const upright = standUpright(cv, [visual, gray], binary)
  visual = ctx.track(upright.images[0])
  gray = ctx.track(upright.images[1])
  binary = ctx.track(upright.binary)

  const { minY, maxY, minX, maxX } = extentsOf(binary)
  const footHeight = Math.max(1, maxY - minY)
  const footWidth = Math.max(1, maxX - minX)

  const bounds: number[] = []
  for (let i = 0; i <= zoneCount; i++) bounds.push(minY + Math.floor((i * footHeight) / zoneCount))
  bounds[zoneCount] = maxY

  const overlay = ctx.track(visual.clone())
  const overlayData = overlay.data as Uint8Array
  const binaryData = binary.data as Uint8Array
  const w = binary.cols

  let totalArea = 0
  for (let i = 0; i < binaryData.length; i++) if (binaryData[i] > 0) totalArea++
  totalArea = Math.max(1, totalArea)

  const metrics: Record<string, number | string> = {}
  const zoneAreas: number[] = []

  for (let z = 0; z < zoneCount; z++) {
    const [name, colour] = zoneDefs[z]
    const y0 = bounds[z]
    const y1 = bounds[z + 1]
    let area = 0
    for (let y = y0; y < y1; y++) {
      const row = y * w
      for (let x = minX; x < maxX; x++) {
        if (binaryData[row + x] === 0) continue
        area++
        const off = (row + x) * 3
        overlayData[off] = colour[2]
        overlayData[off + 1] = colour[1]
        overlayData[off + 2] = colour[0]
      }
    }
    zoneAreas.push(area)
    const percentage = round((100 * area) / totalArea, 1)
    metrics[`${name.toLowerCase()}_area_pct`] = percentage

    const fontScale = Math.max(0.3, (0.55 * footWidth) / 200)
    cv.putText(overlay, `${name}: ${percentage}%`, new cv.Point(minX + 6, Math.floor((y0 + y1) / 2)), cv.FONT_HERSHEY_SIMPLEX, fontScale, new cv.Scalar(255, 255, 255, 255), Math.max(1, Math.floor(footWidth / 150)), cv.LINE_AA)
    if (z < zoneCount - 1) cv.line(overlay, new cv.Point(minX, y1), new cv.Point(maxX, y1), new cv.Scalar(200, 200, 200, 255), 1)
  }

  // Staheli index = arch area over the non-heel area; it describes a plantar arch
  // and is meaningless on anything but a bare print.
  const archArea = zoneCount === 4 ? zoneAreas[2] : zoneAreas[1]
  const nonHeel = zoneCount === 4 ? Math.max(1, zoneAreas[1] + zoneAreas[2]) : Math.max(1, zoneAreas[0] + zoneAreas[1])
  const staheli = round(archArea / nonHeel, 3)
  const archType = staheli < 0.21 ? 'Cavus' : staheli < 0.26 ? 'Normal' : 'Flat'
  metrics.staheli_arch_index = staheli
  metrics.arch_type = archType

  const midX = Math.floor((minX + maxX) / 2)
  let leftArea = 0
  let rightArea = 0
  for (let y = 0; y < binary.rows; y++) {
    const row = y * w
    for (let x = minX; x < midX; x++) if (binaryData[row + x] > 0) leftArea++
    for (let x = midX; x < maxX; x++) if (binaryData[row + x] > 0) rightArea++
  }
  const asymmetry = round(Math.abs(leftArea - rightArea) / Math.max(leftArea + rightArea, 1), 3)
  metrics.asymmetry_score = asymmetry

  let centroidX: number
  let centroidY: number
  if (params.pressure_weights !== false) {
    const grayData = gray.data as Uint8Array
    let weight = 0
    let sumX = 0
    let sumY = 0
    for (let y = 0; y < binary.rows; y++) {
      const row = y * w
      for (let x = 0; x < w; x++) {
        if (binaryData[row + x] === 0) continue
        const value = grayData[row + x]
        weight += value
        sumX += x * value
        sumY += y * value
      }
    }
    centroidX = weight > 0 ? Math.round(sumX / weight) : Math.round((minX + maxX) / 2)
    centroidY = weight > 0 ? Math.round(sumY / weight) : Math.round((minY + maxY) / 2)
  } else {
    centroidX = Math.round((minX + maxX) / 2)
    centroidY = Math.round((minY + maxY) / 2)
  }

  metrics.centroid_x_pct = round((100 * (centroidX - minX)) / footWidth, 1)
  metrics.centroid_y_pct = round((100 * (centroidY - minY)) / footHeight, 1)
  metrics.total_area_px = totalArea
  if (hasCalibration) {
    metrics.foot_length_mm = round(footHeight / pxPerMm, 1)
    metrics.foot_width_mm = round(footWidth / pxPerMm, 1)
  }

  // CBW and HBW are MAXIMUM breadths within their zone, so scan for the widest row.
  const zoneNames = zoneDefs.map((z) => z[0])
  const widths: Record<string, number> = {}
  for (const [key, zoneName] of [
    ['forefoot', 'Forefoot'],
    ['heel', 'Heel'],
  ] as const) {
    const zoneIndex = zoneNames.indexOf(zoneName)
    if (zoneIndex < 0) continue
    let widestRow = -1
    let widestCount = -1
    for (let y = bounds[zoneIndex]; y < bounds[zoneIndex + 1]; y++) {
      const row = y * w
      let count = 0
      for (let x = minX; x < maxX; x++) if (binaryData[row + x] > 0) count++
      if (count > widestCount) {
        widestCount = count
        widestRow = y
      }
    }
    if (widestCount <= 0) continue

    const row = widestRow * w
    let left = -1
    let right = -1
    for (let x = minX; x < maxX; x++) {
      if (binaryData[row + x] === 0) continue
      if (left < 0) left = x
      right = x
    }
    const widthPx = right - left
    widths[key] = widthPx
    metrics[`${key}_width_px`] = widthPx
    if (hasCalibration) metrics[`${key}_width_mm`] = round(widthPx / pxPerMm, 1)

    if (params.show_measurements !== false) {
      cv.line(overlay, new cv.Point(left, widestRow), new cv.Point(right, widestRow), new cv.Scalar(0, 165, 255, 255), Math.max(1, Math.floor(footWidth / 200)))
      const label = hasCalibration ? `${(widthPx / pxPerMm).toFixed(1)}mm` : `${widthPx}px`
      cv.putText(overlay, label, new cv.Point(right + 4, widestRow), cv.FONT_HERSHEY_SIMPLEX, Math.max(0.25, (0.38 * footWidth) / 200), new cv.Scalar(0, 165, 255, 255), 1, cv.LINE_AA)
    }
  }
  if (widths.forefoot > 0) metrics.heel_forefoot_ratio = round((widths.heel ?? 0) / widths.forefoot, 3)

  cv.line(overlay, new cv.Point(midX, minY), new cv.Point(midX, maxY), new cv.Scalar(220, 220, 0, 255), Math.max(1, Math.floor(footWidth / 120)))
  const dotRadius = Math.max(5, Math.floor(footWidth / 50))
  cv.circle(overlay, new cv.Point(centroidX, centroidY), dotRadius, new cv.Scalar(0, 255, 255, 255), -1)
  cv.circle(overlay, new cv.Point(centroidX, centroidY), dotRadius, new cv.Scalar(0, 0, 0, 255), 2)

  const annotationScale = Math.max(0.3, (0.45 * footWidth) / 200)
  const annotationThickness = Math.max(1, Math.floor(footWidth / 180))
  const lineHeight = Math.round((18 * footWidth) / 200)
  cv.putText(overlay, `Staheli ${staheli} - ${archType}`, new cv.Point(minX + 5, maxY - 8), cv.FONT_HERSHEY_SIMPLEX, annotationScale, new cv.Scalar(100, 255, 255, 255), annotationThickness, cv.LINE_AA)
  cv.putText(overlay, `Asym ${asymmetry}`, new cv.Point(minX + 5, maxY - 8 - lineHeight), cv.FONT_HERSHEY_SIMPLEX, annotationScale, new cv.Scalar(220, 220, 0, 255), annotationThickness, cv.LINE_AA)

  const alpha = Number(params.alpha) ?? 0.55
  const blended = ctx.track(new cv.Mat())
  cv.addWeighted(overlay, alpha, visual, 1 - alpha, 0, blended)

  ctx.emit('report', metrics)
  return { main: blended, report: metrics, staheli, asymmetry }
}

function round(value: number, digits: number): number {
  const factor = 10 ** digits
  return Math.round(value * factor) / factor
}
