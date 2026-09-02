import type { NodeImpl } from '../types'
import { colorizeLabels, computeLabelStats, parseColor, toBgr, toGray } from '../cvUtils'

export const featThresholdAdv: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, mask: null }
  const cv = ctx.cv
  const gray = ctx.track(toGray(cv, src))
  const dst = ctx.track(new cv.Mat())
  const mode = Number(params.mode) || 0
  const value = Number(params.threshold) || 127

  if (mode === 0) cv.threshold(gray, dst, value, 255, cv.THRESH_BINARY)
  else if (mode === 1) cv.threshold(gray, dst, value, 255, cv.THRESH_BINARY_INV)
  else if (mode === 2) cv.threshold(gray, dst, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
  else if (mode === 3) cv.threshold(gray, dst, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
  else {
    const data = gray.data as Uint8Array
    let max = 0
    for (let i = 0; i < data.length; i++) if (data[i] > max) max = data[i]
    cv.threshold(gray, dst, 0.7 * max, 255, cv.THRESH_BINARY)
  }
  return { main: dst, mask: dst }
}

export const featMorphologyAdv: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.mask as any
  if (!src) return { main: null, mask: null }
  const cv = ctx.cv
  const shapes = [cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE]
  const size = Math.max(1, Number(params.size) || 5)
  const kernel = cv.getStructuringElement(shapes[Number(params.shape) || 0], new cv.Size(size, size))
  const dst = ctx.track(new cv.Mat())
  const anchor = new cv.Point(-1, -1)
  const iterations = Math.max(1, Number(params.iterations) || 1)
  const op = Number(params.operation) || 0

  if (op === 5) cv.dilate(src, dst, kernel, anchor, iterations, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue())
  else if (op === 6) cv.erode(src, dst, kernel, anchor, iterations, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue())
  else {
    const ops = [cv.MORPH_OPEN, cv.MORPH_CLOSE, cv.MORPH_GRADIENT, cv.MORPH_TOPHAT, cv.MORPH_BLACKHAT]
    cv.morphologyEx(src, dst, ops[op], kernel, anchor, iterations, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue())
  }
  kernel.delete()
  return { main: dst, mask: dst }
}

export const featDistanceTransform: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.mask as any
  if (!src) return { main: null, dist_map: null }
  const cv = ctx.cv
  const binary = ctx.track(toGray(cv, src))
  const types = [cv.DIST_L2, cv.DIST_L1, cv.DIST_C]
  const sizes = [3, 5, cv.DIST_MASK_PRECISE]

  const raw = ctx.track(new cv.Mat())
  cv.distanceTransform(binary, raw, types[Number(params.dist_type) || 0], sizes[Number(params.mask_size) ?? 1])

  const normalized = ctx.track(new cv.Mat())
  cv.normalize(raw, normalized, 0, 255, cv.NORM_MINMAX)
  const asByte = ctx.track(new cv.Mat())
  normalized.convertTo(asByte, cv.CV_8U)

  return { main: asByte, dist_map: raw }
}

export const featConnectedComponents: NodeImpl = (inputs, _params, ctx) => {
  const src = inputs.mask as any
  if (!src) return { main: null, markers: null, count: 0 }
  const cv = ctx.cv
  const binary = ctx.track(toGray(cv, src))
  const labels = ctx.track(new cv.Mat())
  const count = cv.connectedComponents(binary, labels, 8, cv.CV_32S)
  const preview = ctx.track(colorizeLabels(cv, labels))
  return { main: preview, markers: labels, count: Math.max(0, count - 1) }
}

export const sciConnectedComponents: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, count: 0, areas: [], centroids: [], labels_map: null, mask_out: null, contour_out: null }
  const cv = ctx.cv

  const gray = ctx.track(toGray(cv, src))
  const binary = ctx.track(new cv.Mat())
  cv.threshold(gray, binary, Number(params.threshold) ?? 128, 255, cv.THRESH_BINARY)

  const labels = ctx.track(new cv.Mat())
  cv.connectedComponents(binary, labels, Number(params.connectivity) === 1 ? 4 : 8, cv.CV_32S)

  const minArea = Number(params.min_area) ?? 50
  const maxArea = Number(params.max_area) ?? 500000
  const stats = computeLabelStats(labels)
  const data = labels.data32S as Int32Array
  for (let i = 0; i < data.length; i++) {
    const label = data[i]
    if (label <= 0) continue
    const s = stats.get(label)
    if (!s || s.area < minArea || s.area > maxArea) data[i] = 0
  }

  const kept = computeLabelStats(labels)
  const areas = [...kept.values()].map((s) => s.area)
  const centroids = [...kept.values()].map((s) => ({ x: s.cx, y: s.cy }))

  const maskOut = ctx.track(new cv.Mat(labels.rows, labels.cols, cv.CV_8U, new cv.Scalar(0)))
  const maskData = maskOut.data as Uint8Array
  for (let i = 0; i < data.length; i++) maskData[i] = data[i] > 0 ? 255 : 0

  const preview = ctx.track(params.colorize === false ? toBgr(cv, maskOut) : colorizeLabels(cv, labels))

  return { main: preview, count: kept.size, areas, centroids, labels_map: labels, mask_out: maskOut, contour_out: maskOut }
}

export const featMarkerFilter: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.markers as any
  if (!src) return { main: null, markers_out: null, count: 0 }
  const cv = ctx.cv

  const usesPercent = Number(params.area_unit) === 1
  const imageArea = src.rows * src.cols
  const toPixels = (v: number) => (usesPercent ? (v / 100) * imageArea : v)
  const minArea = toPixels(Number(params.min_area) || 0)
  const maxArea = toPixels(Number(params.max_area) || 100000)

  const stats = computeLabelStats(src)
  const remap = new Map<number, number>()
  let next = 1
  for (const [id, s] of stats) {
    if (s.area < minArea || s.area > maxArea) continue
    remap.set(id, Number(params.remap_ids) === 0 ? id : next++)
  }

  const dst = ctx.track(new cv.Mat())
  src.copyTo(dst)
  const data = dst.data32S as Int32Array
  for (let i = 0; i < data.length; i++) {
    const label = data[i]
    if (label <= 0) continue
    data[i] = remap.get(label) ?? 0
  }

  const preview = ctx.track(colorizeLabels(cv, dst))
  return { main: preview, markers_out: dst, count: remap.size }
}

const WATERSHED_BACKGROUND = 1

export const featWatershed: NodeImpl = (inputs, params, ctx) => {
  const image = inputs.image as any
  const seeds = inputs.markers as any
  const cellMask = inputs.mask as any
  if (!image || !seeds) return { main: image ?? null, markers_out: null, count: 0 }
  const cv = ctx.cv

  const bgr = ctx.track(toBgr(cv, image))

  // Reserve label 1 for the background basin, so seed ids shift up by one.
  const markers = ctx.track(new cv.Mat(seeds.rows, seeds.cols, cv.CV_32S, new cv.Scalar(0)))
  const seedData = seeds.data32S as Int32Array
  const markerData = markers.data32S as Int32Array
  let maxSeed = 0
  for (let i = 0; i < seedData.length; i++) {
    if (seedData[i] > 0) {
      markerData[i] = seedData[i] + 1
      if (seedData[i] > maxSeed) maxSeed = seedData[i]
    }
  }

  let maskGray: any = null
  if (cellMask) {
    maskGray = ctx.track(toGray(cv, cellMask))
    const maskData = maskGray.data as Uint8Array
    // Everything outside the cell mask is background, which is what stops the flood.
    for (let i = 0; i < maskData.length; i++) if (maskData[i] === 0) markerData[i] = WATERSHED_BACKGROUND
  }

  if (Number(params.rescue_unseeded) === 1 && maskGray) {
    const blobs = ctx.track(new cv.Mat())
    cv.connectedComponents(maskGray, blobs, 8, cv.CV_32S)
    const blobData = blobs.data32S as Int32Array
    const seeded = new Set<number>()
    for (let i = 0; i < blobData.length; i++) {
      if (blobData[i] > 0 && markerData[i] > WATERSHED_BACKGROUND) seeded.add(blobData[i])
    }
    const rescueMinArea = Number(params.rescue_min_area) || 200
    let nextLabel = maxSeed + 2
    for (const [blobId, s] of computeLabelStats(blobs)) {
      if (seeded.has(blobId) || s.area < rescueMinArea) continue
      markerData[Math.round(s.cy) * blobs.cols + Math.round(s.cx)] = nextLabel++
    }
  }

  cv.watershed(bgr, markers)

  // Renumber to a plain 1..n label map: watershed writes -1 on ridges and 1 on background.
  const finalData = markers.data32S as Int32Array
  const remap = new Map<number, number>()
  let next = 1
  for (let i = 0; i < finalData.length; i++) {
    const v = finalData[i]
    if (v <= WATERSHED_BACKGROUND) {
      finalData[i] = 0
      continue
    }
    let mapped = remap.get(v)
    if (mapped === undefined) {
      mapped = next++
      remap.set(v, mapped)
    }
    finalData[i] = mapped
  }

  const visualization = Number(params.visualization) || 0
  const colorized = ctx.track(colorizeLabels(cv, markers))
  let preview = colorized
  if (visualization === 0 || visualization === 2) {
    const base = visualization === 0 ? bgr : colorized
    preview = ctx.track(base.clone())
    const boundary = parseColor(cv, String(params.boundary_color ?? '#FF0000'), [255, 0, 0])
    const thickness = Math.max(1, Number(params.boundary_thickness) || 1)
    const pixels = preview.data as Uint8Array
    const w = markers.cols
    // A pixel whose right or bottom neighbour has a different label sits on a ridge.
    for (let y = 0; y < markers.rows - 1; y++) {
      for (let x = 0; x < w - 1; x++) {
        const i = y * w + x
        const label = finalData[i]
        if (label === finalData[i + 1] && label === finalData[i + w]) continue
        for (let dy = 0; dy < thickness; dy++) {
          for (let dx = 0; dx < thickness; dx++) {
            const py = y + dy
            const px = x + dx
            if (py >= markers.rows || px >= w) continue
            const off = (py * w + px) * 3
            pixels[off] = boundary[0]
            pixels[off + 1] = boundary[1]
            pixels[off + 2] = boundary[2]
          }
        }
      }
    }
  } else if (visualization === 3) {
    preview = bgr
  }

  return { main: preview, markers_out: markers, count: remap.size }
}
