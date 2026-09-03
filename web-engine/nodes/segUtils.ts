import type { NodeImpl } from '../types'
import { colorizeLabels, toGray } from '../cvUtils'
import { applyColormap, COLORMAPS } from '../colormaps'

function toBinaryGray(cv: any, ctx: any, mask: any): any {
  return ctx.track(toGray(cv, mask))
}

// ---------------------------------------------------------------------------
// Boundary Seeds
// ---------------------------------------------------------------------------
export const featSeedsFromBoundaries: NodeImpl = (inputs, params, ctx) => {
  const mask = inputs.mask as any
  if (!mask) return { markers: null, main: null, dist_map: null, count: 0 }
  const cv = ctx.cv
  const gray = toBinaryGray(cv, ctx, mask)

  const invert = !!params.invert_mask
  let procMask = gray
  if (!invert) {
    const inverted = ctx.track(new cv.Mat())
    cv.bitwise_not(gray, inverted)
    procMask = inverted
  }

  const dist = ctx.track(new cv.Mat())
  cv.distanceTransform(procMask, dist, cv.DIST_L2, 5)
  const distData = dist.data32F as Float32Array
  let maxDist = 0
  for (let i = 0; i < distData.length; i++) if (distData[i] > maxDist) maxDist = distData[i]
  if (maxDist <= 0) return { markers: null, main: null, dist_map: null, count: 0 }

  const peakThreshPct = (Number(params.threshold) || 50) / 100
  const peaks = ctx.track(new cv.Mat())
  cv.threshold(dist, peaks, peakThreshPct * maxDist, 255, cv.THRESH_BINARY)
  const peaksU8 = ctx.track(new cv.Mat())
  peaks.convertTo(peaksU8, cv.CV_8U)

  let peaksFinal = peaksU8
  const dil = Math.round(Number(params.dilation) || 2)
  if (dil > 0) {
    const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(dil * 2 + 1, dil * 2 + 1))
    const dilated = ctx.track(new cv.Mat())
    cv.dilate(peaksFinal, dilated, kernel)
    kernel.delete()
    peaksFinal = dilated
  }

  const markers = ctx.track(new cv.Mat())
  const count = cv.connectedComponents(peaksFinal, markers, 8, cv.CV_32S)

  const colored = ctx.track(colorizeLabels(cv, markers))
  const markerData = markers.data32S as Int32Array
  const coloredData = colored.data as Uint8Array
  for (let i = 0; i < markerData.length; i++) {
    if (markerData[i] === 0) {
      coloredData[i * 3] = 0
      coloredData[i * 3 + 1] = 0
      coloredData[i * 3 + 2] = 0
    }
  }

  const distNorm = ctx.track(new cv.Mat())
  cv.normalize(dist, distNorm, 0, 255, cv.NORM_MINMAX)
  const distVis = ctx.track(new cv.Mat())
  distNorm.convertTo(distVis, cv.CV_8U)

  return { markers, main: colored, dist_map: distVis, count: count - 1 }
}

// ---------------------------------------------------------------------------
// General Segmenter
// ---------------------------------------------------------------------------
export const sciGeneralSegmenter: NodeImpl = (inputs, params, ctx) => {
  const img = inputs.image as any
  if (!img) return { main: null, mask: null, labels: null, count: 0 }
  const cv = ctx.cv

  const bgr = ctx.track(new cv.Mat())
  if (img.channels() === 1) cv.cvtColor(img, bgr, cv.COLOR_GRAY2BGR)
  else img.copyTo(bgr)

  let gray = ctx.track(toGray(cv, bgr))
  if (params.contrast !== false) {
    const clahe = new cv.CLAHE(2.0, new cv.Size(8, 8))
    const out = ctx.track(new cv.Mat())
    clahe.apply(gray, out)
    clahe.delete()
    gray = out
  }

  const sm = Math.round(Number(params.smoothing) || 3)
  if (sm > 0) {
    const blurred = ctx.track(new cv.Mat())
    cv.GaussianBlur(gray, blurred, new cv.Size(sm * 2 + 1, sm * 2 + 1), 0)
    gray = blurred
  }

  const boundaryVal = Math.round(Number(params.boundary) || 5)
  let thresh: any
  if (boundaryVal > 0) {
    const bMask = ctx.track(new cv.Mat())
    cv.threshold(gray, bMask, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    const minLen = boundaryVal * 5
    const lines = ctx.track(new cv.Mat())
    cv.HoughLinesP(bMask, lines, 1, Math.PI / 180, 50, minLen, minLen / 2)
    const straightMask = ctx.track(new cv.Mat(gray.rows, gray.cols, cv.CV_8U, new cv.Scalar(0)))
    const lineData = lines.data32S as Int32Array
    for (let i = 0; i < lineData.length / 4; i++) {
      cv.line(
        straightMask,
        new cv.Point(lineData[i * 4], lineData[i * 4 + 1]),
        new cv.Point(lineData[i * 4 + 2], lineData[i * 4 + 3]),
        new cv.Scalar(255),
        3
      )
    }
    const otsuThresh = ctx.track(new cv.Mat())
    cv.threshold(gray, otsuThresh, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    const notStraight = ctx.track(new cv.Mat())
    cv.bitwise_not(straightMask, notStraight)
    const combined = ctx.track(new cv.Mat())
    cv.bitwise_and(otsuThresh, notStraight, combined)
    const kernelFill = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3))
    const closed = ctx.track(new cv.Mat())
    cv.morphologyEx(combined, closed, cv.MORPH_CLOSE, kernelFill)
    kernelFill.delete()
    thresh = closed
  } else {
    const otsuThresh = ctx.track(new cv.Mat())
    cv.threshold(gray, otsuThresh, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    thresh = otsuThresh
  }

  const externalMarkers = inputs.markers as any
  let markers: any
  let num: number
  if (externalMarkers) {
    markers = ctx.track(new cv.Mat())
    externalMarkers.convertTo(markers, cv.CV_32S)
    const markerData = markers.data32S as Int32Array
    let maxV = 0
    for (let i = 0; i < markerData.length; i++) if (markerData[i] > maxV) maxV = markerData[i]
    num = maxV + 1
  } else {
    const dist = ctx.track(new cv.Mat())
    cv.distanceTransform(thresh, dist, cv.DIST_L2, 5)
    const distBlurred = ctx.track(new cv.Mat())
    cv.GaussianBlur(dist, distBlurred, new cv.Size(5, 5), 0)
    const distData = distBlurred.data32F as Float32Array
    let maxDist = 0
    for (let i = 0; i < distData.length; i++) if (distData[i] > maxDist) maxDist = distData[i]
    const sens = (Number(params.sensitivity) || 30) / 100
    const peaks = ctx.track(new cv.Mat())
    cv.threshold(distBlurred, peaks, sens * maxDist, 255, cv.THRESH_BINARY)
    const peaksU8 = ctx.track(new cv.Mat())
    peaks.convertTo(peaksU8, cv.CV_8U)
    markers = ctx.track(new cv.Mat())
    num = cv.connectedComponents(peaksU8, markers, 8, cv.CV_32S)
  }

  const bgrForWatershed = ctx.track(bgr.clone())
  cv.watershed(bgrForWatershed, markers)

  const minSize = Math.round(Number(params.min_size) || 50)
  const markerData = markers.data32S as Int32Array
  let count: number
  let finalMarkers = markers
  if (minSize > 0) {
    const areas = new Map<number, number>()
    for (let i = 0; i < markerData.length; i++) {
      const v = markerData[i]
      if (v > 0) areas.set(v, (areas.get(v) ?? 0) + 1)
    }
    const validIds = [...areas.entries()].filter(([, area]) => area >= minSize).map(([id]) => id)
    const remap = new Map<number, number>()
    validIds.forEach((id, i) => remap.set(id, i + 1))
    const newMarkers = ctx.track(new cv.Mat(markers.rows, markers.cols, cv.CV_32S, new cv.Scalar(0)))
    const newData = newMarkers.data32S as Int32Array
    for (let i = 0; i < markerData.length; i++) {
      const mapped = remap.get(markerData[i])
      if (mapped !== undefined) newData[i] = mapped
    }
    finalMarkers = newMarkers
    count = validIds.length
  } else {
    count = num - 1
  }

  const finalData = finalMarkers.data32S as Int32Array
  const vizMode = Number(params.viz_mode) ?? 2
  const n = Math.max(1, count)
  const visStones = ctx.track(new cv.Mat(finalMarkers.rows, finalMarkers.cols, cv.CV_8U))
  const visData = visStones.data as Uint8Array
  for (let i = 0; i < finalData.length; i++) visData[i] = Math.max(0, Math.min(255, Math.round((Math.max(0, finalData[i]) * 255) / n)))
  const colored = ctx.track(applyColormap(cv, visStones, COLORMAPS.Turbo))
  const coloredData = colored.data as Uint8Array
  for (let i = 0; i < finalData.length; i++) {
    if (finalData[i] <= 0) {
      coloredData[i * 3] = 0
      coloredData[i * 3 + 1] = 0
      coloredData[i * 3 + 2] = 0
    }
  }

  const res = ctx.track(bgr.clone())
  const resData = res.data as Uint8Array
  for (let i = 0; i < finalData.length; i++) {
    if (finalData[i] === -1) {
      resData[i * 3] = 0
      resData[i * 3 + 1] = 0
      resData[i * 3 + 2] = 255
    }
  }

  let out: any
  if (vizMode === 0) {
    out = res
  } else if (vizMode === 1) {
    const blended = ctx.track(new cv.Mat())
    cv.addWeighted(bgr, 0.4, colored, 0.6, 0, blended)
    out = blended
  } else {
    const blended = ctx.track(new cv.Mat())
    cv.addWeighted(bgr, 0.4, colored, 0.6, 0, blended)
    const blendedData = blended.data as Uint8Array
    for (let i = 0; i < finalData.length; i++) {
      if (finalData[i] === -1) {
        blendedData[i * 3] = 0
        blendedData[i * 3 + 1] = 0
        blendedData[i * 3 + 2] = 255
      }
    }
    out = blended
  }

  const maskOut = ctx.track(new cv.Mat(gray.rows, gray.cols, cv.CV_8U, new cv.Scalar(0)))
  const maskData = maskOut.data as Uint8Array
  for (let i = 0; i < finalData.length; i++) if (finalData[i] > 0) maskData[i] = 255

  const maskThick = Math.round(Number(params.mask_thick) || 1)
  let maskFinal = maskOut
  if (maskThick > 0) {
    const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(maskThick * 2 + 1, maskThick * 2 + 1))
    const eroded = ctx.track(new cv.Mat())
    cv.erode(maskOut, eroded, kernel)
    kernel.delete()
    maskFinal = eroded
  }

  return { main: out, mask: maskFinal, labels: finalMarkers, count }
}

// ---------------------------------------------------------------------------
// Linearity Filter
// ---------------------------------------------------------------------------
export const filterLinearity: NodeImpl = (inputs, params, ctx) => {
  const img = inputs.image as any
  if (!img) return { main: null, mask: null }
  const cv = ctx.cv
  const gray = ctx.track(toGray(cv, img))

  const mask = ctx.track(new cv.Mat())
  cv.threshold(gray, mask, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

  const minLen = Math.round(Number(params.min_length) || 25)
  const thresh = Math.round(Number(params.threshold) || 50)
  const thick = Math.round(Number(params.thickness) || 3)

  const lines = ctx.track(new cv.Mat())
  cv.HoughLinesP(mask, lines, 1, Math.PI / 180, thresh, minLen, minLen / 2)

  const outMask = ctx.track(new cv.Mat(gray.rows, gray.cols, cv.CV_8U, new cv.Scalar(0)))
  const lineData = lines.data32S as Int32Array
  for (let i = 0; i < lineData.length / 4; i++) {
    cv.line(outMask, new cv.Point(lineData[i * 4], lineData[i * 4 + 1]), new cv.Point(lineData[i * 4 + 2], lineData[i * 4 + 3]), new cv.Scalar(255), thick)
  }

  const main = ctx.track(new cv.Mat())
  cv.cvtColor(outMask, main, cv.COLOR_GRAY2BGR)
  return { main, mask: outMask }
}

// ---------------------------------------------------------------------------
// Region Sealer
// ---------------------------------------------------------------------------
export const maskRegionSealer: NodeImpl = (inputs, params, ctx) => {
  const mask = inputs.mask as any
  if (!mask) return { mask: null }
  const cv = ctx.cv
  const gray = toBinaryGray(cv, ctx, mask)

  const tol = Math.max(1, Math.round(Number(params.gap_tolerance) || 5))
  const fill = params.fill_regions !== false
  const minArea = Number(params.min_area) || 100

  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(tol, tol))
  const closed = ctx.track(new cv.Mat())
  cv.morphologyEx(gray, closed, cv.MORPH_CLOSE, kernel)
  kernel.delete()

  if (!fill) return { mask: closed }

  const inv = ctx.track(new cv.Mat())
  cv.bitwise_not(closed, inv)
  const contours = new cv.MatVector()
  const hierarchy = ctx.track(new cv.Mat())
  cv.findContours(inv, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

  const filled = ctx.track(new cv.Mat(gray.rows, gray.cols, cv.CV_8U, new cv.Scalar(0)))
  for (let i = 0; i < contours.size(); i++) {
    const cnt = contours.get(i)
    if (cv.contourArea(cnt) >= minArea) {
      const single = new cv.MatVector()
      single.push_back(cnt)
      cv.drawContours(filled, single, -1, new cv.Scalar(255), -1)
      single.delete()
    }
  }
  contours.delete()

  return { mask: filled }
}

// ---------------------------------------------------------------------------
// Direction Filter
// ---------------------------------------------------------------------------
export const filterLinearDirection: NodeImpl = (inputs, params, ctx) => {
  const img = inputs.image as any
  if (!img) return { main: null, mask: null }
  const cv = ctx.cv
  const gray = ctx.track(toGray(cv, img))

  const binary = ctx.track(new cv.Mat())
  cv.threshold(gray, binary, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

  const targetAngle = Number(params.angle) || 0
  const tolerance = Number(params.tolerance) || 10
  const minLen = Math.round(Number(params.min_length) || 20)
  const extend = Math.round(Number(params.extend) || 0)
  const thick = Math.round(Number(params.thickness) || 5)
  const mode = Number(params.mode) || 0

  const lines = ctx.track(new cv.Mat())
  cv.HoughLinesP(binary, lines, 1, Math.PI / 180, 30, minLen, minLen / 2)

  const maskOverlay = ctx.track(new cv.Mat(gray.rows, gray.cols, cv.CV_8U, new cv.Scalar(0)))
  const lineData = lines.data32S as Int32Array
  for (let i = 0; i < lineData.length / 4; i++) {
    let x1 = lineData[i * 4], y1 = lineData[i * 4 + 1], x2 = lineData[i * 4 + 2], y2 = lineData[i * 4 + 3]
    const angle = (((Math.atan2(y2 - y1, x2 - x1) * 180) / Math.PI) % 180 + 180) % 180
    let diff = Math.abs(angle - targetAngle)
    if (diff > 90) diff = 180 - diff
    const match = diff <= tolerance

    if (extend > 0) {
      const dx = x2 - x1, dy = y2 - y1
      const dist = Math.hypot(dx, dy)
      if (dist > 0) {
        const ex = (dx / dist) * extend
        const ey = (dy / dist) * extend
        x1 = Math.round(x1 - ex)
        y1 = Math.round(y1 - ey)
        x2 = Math.round(x2 + ex)
        y2 = Math.round(y2 + ey)
      }
    }

    if (match) cv.line(maskOverlay, new cv.Point(x1, y1), new cv.Point(x2, y2), new cv.Scalar(255), thick)
  }

  let resMask: any
  if (mode === 0) {
    resMask = ctx.track(new cv.Mat())
    cv.subtract(binary, maskOverlay, resMask)
  } else {
    resMask = maskOverlay
  }

  const main = ctx.track(new cv.Mat())
  cv.cvtColor(resMask, main, cv.COLOR_GRAY2BGR)
  return { main, mask: resMask }
}

// ---------------------------------------------------------------------------
// Anisotropic Morphology
// ---------------------------------------------------------------------------
export const filterDirectionalMorphology: NodeImpl = (inputs, params, ctx) => {
  const mask = inputs.mask as any
  if (!mask) return { mask: null }
  const cv = ctx.cv
  const gray = toBinaryGray(cv, ctx, mask)

  const axis = Number(params.axis) || 0
  const opIdx = Number(params.operation) || 0
  const size = Math.max(1, Math.round(Number(params.size) || 5))

  const kernel = axis === 0 ? cv.matFromArray(1, size, cv.CV_8U, new Array(size).fill(1)) : cv.matFromArray(size, 1, cv.CV_8U, new Array(size).fill(1))

  const res = ctx.track(new cv.Mat())
  if (opIdx === 0) cv.dilate(gray, res, kernel)
  else if (opIdx === 1) cv.erode(gray, res, kernel)
  else if (opIdx === 2) cv.morphologyEx(gray, res, cv.MORPH_OPEN, kernel)
  else cv.morphologyEx(gray, res, cv.MORPH_CLOSE, kernel)
  kernel.delete()

  return { mask: res }
}
