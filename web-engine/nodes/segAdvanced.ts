import type { NodeImpl } from '../types'
import { toBgr, toGray } from '../cvUtils'
import { applyColormap, infernoColor, jetColor, oceanColor, viridisColor } from '../colormaps'

/** #rrggbb → BGR. */
function hexToBgr(raw: unknown, fallback: [number, number, number]): [number, number, number] {
  let hex = String(raw ?? '').trim().replace(/^#/, '')
  if (hex.length === 3) hex = hex.split('').map((c) => c + c).join('')
  if (!/^[0-9a-fA-F]{6}$/.test(hex)) return fallback
  return [parseInt(hex.slice(4, 6), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(0, 2), 16)]
}

/* --------------------------------------------------------------- blob filter */

export const filterBlobFilter: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const src = (inputs.mask ?? inputs.main) as any
  if (!src) return { main: null, mask: null, count: 0 }

  let gray = toGray(cv, src)
  if (gray.depth() !== cv.CV_8U) {
    // A float mask runs 0–1; anything else is clamped into the byte range.
    const data = gray.depth() === cv.CV_32F ? gray.data32F : gray.data64F
    let peak = 0
    for (let i = 0; i < data.length; i++) if (data[i] > peak) peak = data[i]
    const eight = new cv.Mat(gray.rows, gray.cols, cv.CV_8U)
    const scale = peak <= 1.01 ? 255 : 1
    for (let i = 0; i < data.length; i++) eight.data[i] = Math.max(0, Math.min(255, Math.round(data[i] * scale)))
    gray.delete()
    gray = eight
  } else {
    // A 0/1 integer mask would be wiped out by the default threshold of 127.
    let peak = 0
    for (let i = 0; i < gray.data.length; i++) if (gray.data[i] > peak) peak = gray.data[i]
    if (peak <= 1) for (let i = 0; i < gray.data.length; i++) gray.data[i] *= 255
  }

  const threshold = Math.max(1, Math.min(254, Math.round(Number(params.threshold) ?? 127)))
  const binary = new cv.Mat()
  cv.threshold(gray, binary, threshold, 255, cv.THRESH_BINARY)
  gray.delete()

  const minArea = Math.max(1, Math.round(Number(params.min_area) ?? 100))
  const maxArea = Math.round(Number(params.max_area) || 0)
  const circMin = Number(params.circ_min) || 0
  const circMax = Number(params.circ_max) || 0
  const elongMin = Number(params.elong_min) ?? 1
  const elongMax = Number(params.elong_max) || 0
  const connectivity = Math.round(Number(params.connectivity) || 8) === 4 ? 4 : 8

  const labels = new cv.Mat()
  const stats = new cv.Mat()
  const centroids = new cv.Mat()
  const count = cv.connectedComponentsWithStats(binary, labels, stats, centroids, connectivity, cv.CV_32S)

  const shapeActive = circMin > 0 || circMax > 0 || elongMin > 1 || elongMax > 0
  const labelData = labels.data32S
  const keep = new Set<number>()

  for (let i = 1; i < count; i++) {
    const area = stats.intAt(i, cv.CC_STAT_AREA)
    if (area < minArea) continue
    if (maxArea > 0 && area > maxArea) continue

    if (shapeActive) {
      const x = stats.intAt(i, cv.CC_STAT_LEFT)
      const y = stats.intAt(i, cv.CC_STAT_TOP)
      const w = stats.intAt(i, cv.CC_STAT_WIDTH)
      const h = stats.intAt(i, cv.CC_STAT_HEIGHT)

      const sub = new cv.Mat(h, w, cv.CV_8U)
      const subData = sub.data
      let m00 = 0
      let m10 = 0
      let m01 = 0
      for (let yy = 0; yy < h; yy++) {
        for (let xx = 0; xx < w; xx++) {
          const on = labelData[(y + yy) * labels.cols + (x + xx)] === i
          subData[yy * w + xx] = on ? 255 : 0
          if (on) {
            m00 += 1
            m10 += xx
            m01 += yy
          }
        }
      }

      const contours = new cv.MatVector()
      const hierarchy = new cv.Mat()
      cv.findContours(sub, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
      let perimeter = 0
      let best = -1
      let bestArea = -1
      for (let c = 0; c < contours.size(); c++) {
        const a = cv.contourArea(contours.get(c))
        if (a > bestArea) {
          bestArea = a
          best = c
        }
      }
      if (best >= 0) perimeter = cv.arcLength(contours.get(best), true)
      contours.delete()
      hierarchy.delete()

      // C = 4πA/P², capped at 1 — discretisation can push it slightly over.
      const circ = perimeter > 0 ? Math.min((4 * Math.PI * area) / (perimeter * perimeter), 1) : 0

      // Elongation from the second-order central moments: √(λ₁/λ₂).
      let elong = 1
      if (m00 > 0) {
        const cx = m10 / m00
        const cy = m01 / m00
        let mu20 = 0
        let mu11 = 0
        let mu02 = 0
        for (let yy = 0; yy < h; yy++) {
          for (let xx = 0; xx < w; xx++) {
            if (!subData[yy * w + xx]) continue
            const dx = xx - cx
            const dy = yy - cy
            mu20 += dx * dx
            mu11 += dx * dy
            mu02 += dy * dy
          }
        }
        const a = mu20 / m00
        const b = mu11 / m00
        const d = mu02 / m00
        const common = Math.sqrt(Math.max((a - d) ** 2 + 4 * b * b, 0))
        const l1 = (a + d + common) / 2
        const l2 = (a + d - common) / 2
        elong = l2 > 1e-9 ? Math.sqrt(l1 / l2) : 1e3
      }
      sub.delete()

      if (circ < circMin) continue
      if (circMax > 0 && circ > circMax) continue
      if (elong < elongMin) continue
      if (elongMax > 0 && elong > elongMax) continue
    }

    keep.add(i)
  }

  const out = ctx.track(new cv.Mat(binary.rows, binary.cols, cv.CV_8U))
  const bits = out.data
  for (let i = 0; i < labelData.length; i++) bits[i] = keep.has(labelData[i]) ? 255 : 0

  labels.delete()
  stats.delete()
  centroids.delete()
  binary.delete()

  ctx.emit('count', keep.size)
  return { main: out, mask: out, count: keep.size }
}

/* -------------------------------------------------------- smart morphology */

export const filterMorphologySmart: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const src = (inputs.mask ?? inputs.main) as any
  if (!src) return { mask: null }

  const gray = toGray(cv, src)
  const binary = new cv.Mat()
  cv.threshold(gray, binary, 127, 255, cv.THRESH_BINARY)
  gray.delete()

  const w = binary.cols
  const h = binary.rows
  const totalArea = w * h
  const thresholdPct = Number(params.area_thresh_pct) ?? 0.5
  const amount = Math.max(1, Math.round(Number(params.amount) || 3))

  const labels = new cv.Mat()
  const stats = new cv.Mat()
  const centroids = new cv.Mat()
  const count = cv.connectedComponentsWithStats(binary, labels, stats, centroids)
  const labelData = labels.data32S

  // Split by size, then shrink the small pieces and grow the large ones — noise
  // erodes away while the real structures gain body.
  const small = new cv.Mat(h, w, cv.CV_8U, new cv.Scalar(0))
  const large = new cv.Mat(h, w, cv.CV_8U, new cv.Scalar(0))
  const isSmall = new Set<number>()
  for (let i = 1; i < count; i++) {
    const areaPct = (stats.intAt(i, cv.CC_STAT_AREA) / totalArea) * 100
    if (areaPct < thresholdPct) isSmall.add(i)
  }
  for (let i = 0; i < labelData.length; i++) {
    const id = labelData[i]
    if (!id) continue
    if (isSmall.has(id)) small.data[i] = 255
    else large.data[i] = 255
  }

  const kernel = cv.Mat.ones(amount, amount, cv.CV_8U)
  const anchor = new cv.Point(-1, -1)
  const erodedSmall = new cv.Mat()
  const dilatedLarge = new cv.Mat()
  cv.erode(small, erodedSmall, kernel, anchor, 1)
  cv.dilate(large, dilatedLarge, kernel, anchor, 1)
  kernel.delete()

  const out = ctx.track(new cv.Mat())
  cv.bitwise_or(erodedSmall, dilatedLarge, out)

  small.delete()
  large.delete()
  erodedSmall.delete()
  dilatedLarge.delete()
  labels.delete()
  stats.delete()
  centroids.delete()
  binary.delete()
  return { mask: out }
}

/* ------------------------------------------------------------ stereo depth */

export const cvStereo: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const left = inputs.left as any
  const right = inputs.right as any
  if (!left || !right) return { main: null, disp_min: 0, disp_max: 0, data: {} }

  const grayL = toGray(cv, left)
  let grayR = toGray(cv, right)
  if (grayR.cols !== grayL.cols || grayR.rows !== grayL.rows) {
    const resized = new cv.Mat()
    cv.resize(grayR, resized, new cv.Size(grayL.cols, grayL.rows), 0, 0, cv.INTER_LINEAR)
    grayR.delete()
    grayR = resized
  }

  // numDisparities must be a positive multiple of 16, blockSize odd.
  const numDisparities = Math.max(16, Math.round((Number(params.num_disparities) || 64) / 16) * 16)
  let blockSize = Math.max(3, Math.min(15, Math.round(Number(params.block_size) || 7)))
  if (blockSize % 2 === 0) blockSize += 1
  const minDisparity = Math.round(Number(params.min_disparity) || 0)

  const w = grayL.cols
  const h = grayL.rows
  const n = w * h
  const leftPixels = grayL.data
  const rightPixels = grayR.data
  const half = blockSize >> 1

  // cv.StereoBM and cv.StereoSGBM are compiled into the WASM but not exposed to
  // JavaScript by this build's embind wrappers, so the block matcher is written
  // out here: for each pixel, the disparity whose window has the smallest sum of
  // absolute differences. That is what StereoBM does; SGBM's semi-global path
  // aggregation is not reproduced, so the map is noisier in low-texture areas.
  const disparity = new Float32Array(n)
  for (let y = half; y < h - half; y++) {
    for (let x = half + numDisparities + minDisparity; x < w - half; x++) {
      let best = 0
      let bestCost = Infinity
      let secondCost = Infinity
      for (let d = minDisparity; d < minDisparity + numDisparities; d++) {
        let cost = 0
        for (let dy = -half; dy <= half; dy++) {
          const rowL = (y + dy) * w
          const rowR = rowL - d
          for (let dx = -half; dx <= half; dx++) {
            cost += Math.abs(leftPixels[rowL + x + dx] - rightPixels[rowR + x + dx])
          }
        }
        if (cost < bestCost) {
          secondCost = bestCost
          bestCost = cost
          best = d
        } else if (cost < secondCost) secondCost = cost
      }
      // A winner no better than the runner-up is not a match worth trusting.
      disparity[y * w + x] = secondCost > 0 && bestCost / secondCost > 0.95 ? 0 : best
    }
  }

  let lo = Infinity
  let hi = -Infinity
  let any = false
  for (let i = 0; i < n; i++) {
    const d = disparity[i]
    // Zero means the pixel found no match; those are not depth.
    if (d > minDisparity) {
      any = true
      if (d < lo) lo = d
      if (d > hi) hi = d
    }
  }
  if (!any) {
    lo = 0
    hi = 0
  }

  let vmin = Infinity
  let vmax = -Infinity
  for (let i = 0; i < n; i++) {
    if (disparity[i] < vmin) vmin = disparity[i]
    if (disparity[i] > vmax) vmax = disparity[i]
  }
  const span = vmax - vmin || 1
  const eight = new cv.Mat(h, w, cv.CV_8U)
  for (let i = 0; i < n; i++) eight.data[i] = Math.round(((disparity[i] - vmin) / span) * 255)

  const choice = String(params.colormap ?? 'Jet')
  let out: any
  if (choice === 'Gray' || Number(params.colormap) === 2) {
    out = ctx.track(new cv.Mat())
    cv.cvtColor(eight, out, cv.COLOR_GRAY2BGR)
  } else {
    const fn = choice === 'Viridis' || Number(params.colormap) === 1 ? viridisColor : jetColor
    out = ctx.track(applyColormap(cv, eight, fn))
  }

  eight.delete()
  grayL.delete()
  grayR.delete()

  return {
    main: out,
    disp_min: Math.round(lo * 100) / 100,
    disp_max: Math.round(hi * 100) / 100,
    data: { num_disparities: numDisparities, block_size: blockSize },
  }
}

/* ----------------------------------------------------------------- GrabCut */

/** Manual-Points list → foreground and background pixel seeds. */
function splitSeeds(pts: unknown, w: number, h: number): { fg: [number, number][]; bg: [number, number][] } {
  const fg: [number, number][] = []
  const bg: [number, number][] = []
  if (!Array.isArray(pts)) return { fg, bg }
  for (const raw of pts) {
    let x: number
    let y: number
    let label = 1
    if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
      const p = raw as Record<string, unknown>
      x = Number(p.x) || 0
      y = Number(p.y) || 0
      label = p.label === undefined ? 1 : Number(p.label)
    } else if (Array.isArray(raw) && raw.length >= 2) {
      x = Number(raw[0])
      y = Number(raw[1])
      if (raw.length > 2) label = Number(raw[2])
    } else continue
    // Coordinates inside the unit square are normalised, larger ones are pixels.
    const px = Math.max(0, Math.min(w - 1, Math.round(x >= 0 && x <= 1 ? x * w : x)))
    const py = Math.max(0, Math.min(h - 1, Math.round(y >= 0 && y <= 1 ? y * h : y)))
    ;(label === 1 ? fg : bg).push([px, py])
  }
  return { fg, bg }
}

export const featGrabcut: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const source = (inputs.image ?? inputs.main) as any
  if (!source) return { main: null, mask: null, cutout: null, count: 0 }

  const img = toBgr(cv, source)
  const w = img.cols
  const h = img.rows

  const render = (fgMask: any) => {
    const opacity = (Number(params.overlay_opacity) ?? 50) / 100
    const colour = hexToBgr(params.fg_color, [85, 221, 34])
    const overlay = ctx.track(img.clone())

    if (opacity > 0) {
      const data = overlay.data
      const bits = fgMask.data
      for (let p = 0, i = 0; p < bits.length; p++, i += 3) {
        if (!bits[p]) continue
        data[i] = data[i] * (1 - opacity) + colour[0] * opacity
        data[i + 1] = data[i + 1] * (1 - opacity) + colour[1] * opacity
        data[i + 2] = data[i + 2] * (1 - opacity) + colour[2] * opacity
      }
    }
    if (params.show_contour !== false) {
      const contours = new cv.MatVector()
      const hierarchy = new cv.Mat()
      cv.findContours(fgMask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
      cv.drawContours(overlay, contours, -1, new cv.Scalar(colour[0], colour[1], colour[2], 255), 2, cv.LINE_AA)
      contours.delete()
      hierarchy.delete()
    }

    const cutoutMode = Math.round(Number(params.cutout_bg) || 0)
    let cutout: any
    if (cutoutMode === 2) {
      cutout = ctx.track(new cv.Mat())
      cv.cvtColor(img, cutout, cv.COLOR_BGR2BGRA)
      const data = cutout.data
      const bits = fgMask.data
      for (let p = 0; p < bits.length; p++) data[p * 4 + 3] = bits[p]
    } else {
      const fill = cutoutMode === 0 ? 0 : 255
      cutout = ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(fill, fill, fill, 255)))
      const data = cutout.data
      const srcData = img.data
      const bits = fgMask.data
      for (let p = 0, i = 0; p < bits.length; p++, i += 3) {
        if (!bits[p]) continue
        data[i] = srcData[i]
        data[i + 1] = srcData[i + 1]
        data[i + 2] = srcData[i + 2]
      }
    }

    const count = cv.countNonZero(fgMask)
    img.delete()
    return { main: overlay, mask: ctx.track(fgMask), cutout, count }
  }

  const hint = (message: string) => {
    const out = ctx.track(img.clone())
    cv.rectangle(out, new cv.Point(0, 0), new cv.Point(w, 30), new cv.Scalar(20, 20, 20, 255), -1)
    cv.putText(out, message, new cv.Point(8, 21), cv.FONT_HERSHEY_SIMPLEX, 0.55, new cv.Scalar(50, 200, 200, 255), 1, cv.LINE_AA)
    img.delete()
    return { main: out, mask: null, cutout: null, count: 0 }
  }

  const state = (ctx.state.get(ctx.nodeId) as { lastRun: number; mask: any } | undefined) ?? { lastRun: 0, mask: null }
  const runTrigger = Number(params.run) ? 1 : 0
  const fired = runTrigger === 1 && state.lastRun === 0
  state.lastRun = runTrigger
  ctx.state.set(ctx.nodeId, state)
  const live = !!params.live

  // GrabCut is far too heavy for 30 fps, so it only runs on demand and the
  // last result is re-rendered in between.
  if (!fired && !live) {
    if (state.mask) return render(state.mask.clone())
    return hint('Press Run')
  }

  const initMode = Math.round(Number(params.init_mode) || 0)
  const iterations = Math.max(1, Math.round(Number(params.iterations) || 5))
  const brush = Math.max(1, Math.round(Number(params.brush) || 14))
  const { fg, bg } = splitSeeds(inputs.points, w, h)

  const gcMask = new cv.Mat(h, w, cv.CV_8U, new cv.Scalar(cv.GC_PR_BGD))
  let rect: any = null

  if (initMode === 0 && fg.length) {
    const margin = (Number(params.rect_margin) ?? 8) / 100
    const xs = fg.map((p) => p[0])
    const ys = fg.map((p) => p[1])
    const mx = Math.round(w * margin)
    const my = Math.round(h * margin)
    const x0 = Math.max(0, Math.min(...xs) - mx)
    const y0 = Math.max(0, Math.min(...ys) - my)
    const x1 = Math.min(w, Math.max(...xs) + mx)
    const y1 = Math.min(h, Math.max(...ys) + my)
    if (x1 > x0 && y1 > y0) rect = new cv.Rect(x0, y0, x1 - x0, y1 - y0)
  }
  if (!rect && (initMode === 0 || initMode === 1)) {
    const pct = (Number(params.center_rect) ?? 70) / 100
    const bw = Math.round(w * pct)
    const bh = Math.round(h * pct)
    rect = new cv.Rect(Math.round((w - bw) / 2), Math.round((h - bh) / 2), bw, bh)
  }

  if (rect) {
    gcMask.setTo(new cv.Scalar(cv.GC_BGD))
    gcMask.roi(rect).setTo(new cv.Scalar(cv.GC_PR_FGD))
  }

  if (initMode === 3 && inputs.mask) {
    let m = toGray(cv, inputs.mask as any)
    if (m.cols !== w || m.rows !== h) {
      const resized = new cv.Mat()
      cv.resize(m, resized, new cv.Size(w, h), 0, 0, cv.INTER_NEAREST)
      m.delete()
      m = resized
    }
    gcMask.setTo(new cv.Scalar(cv.GC_PR_BGD))
    const bits = m.data
    const gc = gcMask.data
    for (let i = 0; i < bits.length; i++) if (bits[i] > 127) gc[i] = cv.GC_PR_FGD
    m.delete()
  }

  // Scribbles are hard seeds and always win over whatever the init produced.
  for (const [px, py] of fg) cv.circle(gcMask, new cv.Point(px, py), brush, new cv.Scalar(cv.GC_FGD), -1)
  for (const [px, py] of bg) cv.circle(gcMask, new cv.Point(px, py), brush, new cv.Scalar(cv.GC_BGD), -1)

  let hasSeed = false
  for (let i = 0; i < gcMask.data.length; i++) {
    const v = gcMask.data[i]
    if (v === cv.GC_FGD || v === cv.GC_PR_FGD) {
      hasSeed = true
      break
    }
  }
  if (!hasSeed) {
    gcMask.delete()
    return hint('Add FG seeds / rect')
  }

  const bgdModel = new cv.Mat()
  const fgdModel = new cv.Mat()
  try {
    cv.grabCut(img, gcMask, rect ?? new cv.Rect(0, 0, 1, 1), bgdModel, fgdModel, iterations, cv.GC_INIT_WITH_MASK)
  } catch {
    bgdModel.delete()
    fgdModel.delete()
    gcMask.delete()
    return hint('GrabCut failed')
  }
  bgdModel.delete()
  fgdModel.delete()

  const fgMask = new cv.Mat(h, w, cv.CV_8U)
  const gc = gcMask.data
  for (let i = 0; i < gc.length; i++) fgMask.data[i] = gc[i] === cv.GC_FGD || gc[i] === cv.GC_PR_FGD ? 255 : 0
  gcMask.delete()

  state.mask?.delete?.()
  state.mask = fgMask.clone()
  return render(fgMask)
}

/* -------------------------------------------------- morphological snakes */

/**
 * The four 3×3 line structuring elements the morphological curvature operator
 * uses: horizontal, vertical and the two diagonals. Offsets are [dy, dx].
 */
const LINE_ELEMENTS: [number, number][][] = [
  [[0, -1], [0, 0], [0, 1]],
  [[-1, 0], [0, 0], [1, 0]],
  [[-1, -1], [0, 0], [1, 1]],
  [[-1, 1], [0, 0], [1, -1]],
]

/**
 * Erosion with a zero border, matching scipy's `binary_erosion` default: a
 * structuring element reaching outside the image sees False there, so the
 * result is 0. Clamping to the edge pixel instead would leave spurious
 * survivors along the border.
 */
function erodeBy(u: Uint8Array, w: number, h: number, element: [number, number][], out: Uint8Array): void {
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let all = 1
      for (const [dy, dx] of element) {
        const yy = y + dy
        const xx = x + dx
        if (yy < 0 || yy >= h || xx < 0 || xx >= w || !u[yy * w + xx]) {
          all = 0
          break
        }
      }
      out[y * w + x] = all
    }
  }
}

/** Dilation with the same zero border convention. */
function dilateBy(u: Uint8Array, w: number, h: number, element: [number, number][], out: Uint8Array): void {
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let any = 0
      for (const [dy, dx] of element) {
        const yy = y + dy
        const xx = x + dx
        if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue
        if (u[yy * w + xx]) {
          any = 1
          break
        }
      }
      out[y * w + x] = any
    }
  }
}

/** SI: the maximum over erosions by each line element. */
function supInf(u: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(u.length)
  const scratch = new Uint8Array(u.length)
  for (let e = 0; e < LINE_ELEMENTS.length; e++) {
    erodeBy(u, w, h, LINE_ELEMENTS[e], scratch)
    for (let i = 0; i < out.length; i++) if (scratch[i] > out[i]) out[i] = scratch[i]
  }
  return out
}

/** IS: the minimum over dilations by each line element. */
function infSup(u: Uint8Array, w: number, h: number): Uint8Array {
  const out = new Uint8Array(u.length).fill(1)
  const scratch = new Uint8Array(u.length)
  for (let e = 0; e < LINE_ELEMENTS.length; e++) {
    dilateBy(u, w, h, LINE_ELEMENTS[e], scratch)
    for (let i = 0; i < out.length; i++) if (scratch[i] < out[i]) out[i] = scratch[i]
  }
  return out
}

/**
 * The morphological curvature operator, alternating SI∘IS and IS∘SI on
 * successive calls exactly as scikit-image does — always applying the same
 * composition would bias the contour in one direction.
 */
function curvatureStep(u: Uint8Array, w: number, h: number, parity: number): Uint8Array {
  return parity % 2 === 0 ? supInf(infSup(u, w, h), w, h) : infSup(supInf(u, w, h), w, h)
}

/** Central-difference gradient with one-sided edges, matching numpy's gradient. */
function gradient(field: Float32Array, w: number, h: number): { dy: Float32Array; dx: Float32Array } {
  const dy = new Float32Array(field.length)
  const dx = new Float32Array(field.length)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x
      dy[i] = h === 1 ? 0 : y === 0 ? field[i + w] - field[i] : y === h - 1 ? field[i] - field[i - w] : (field[i + w] - field[i - w]) / 2
      dx[i] = w === 1 ? 0 : x === 0 ? field[i + 1] - field[i] : x === w - 1 ? field[i] - field[i - 1] : (field[i + 1] - field[i - 1]) / 2
    }
  }
  return { dy, dx }
}

/**
 * Morphological Chan–Vese: the level set moves so that the image means inside
 * and outside the contour are best separated. No edges are needed, which is why
 * it copes with blurred or broken boundaries.
 */
function morphChanVese(image: Float32Array, u: Uint8Array, w: number, h: number, iterations: number, smoothing: number, lambda1: number, lambda2: number): Uint8Array {
  let level = u
  for (let it = 0; it < iterations; it++) {
    let sumIn = 0
    let countIn = 0
    let sumOut = 0
    let countOut = 0
    for (let i = 0; i < level.length; i++) {
      if (level[i]) {
        sumIn += image[i]
        countIn++
      } else {
        sumOut += image[i]
        countOut++
      }
    }
    if (!countIn || !countOut) break
    const c1 = sumIn / countIn
    const c0 = sumOut / countOut

    const asFloat = new Float32Array(level.length)
    for (let i = 0; i < level.length; i++) asFloat[i] = level[i]
    const { dy, dx } = gradient(asFloat, w, h)

    const next = new Uint8Array(level)
    for (let i = 0; i < level.length; i++) {
      const absDu = Math.abs(dy[i]) + Math.abs(dx[i])
      const aux = absDu * (lambda1 * (image[i] - c1) ** 2 - lambda2 * (image[i] - c0) ** 2)
      if (aux < 0) next[i] = 1
      else if (aux > 0) next[i] = 0
    }
    level = next
    for (let s = 0; s < smoothing; s++) level = curvatureStep(level, w, h, it + s)
  }
  return level
}

/**
 * Morphological geodesic active contour: the level set is pushed by a balloon
 * force and stopped where the edge-indicator image is small — the classic
 * edge-driven snake, done with morphology instead of PDEs.
 */
function morphGac(gimage: Float32Array, u: Uint8Array, w: number, h: number, iterations: number, smoothing: number, balloon: number, threshold: number): Uint8Array {
  let level = u
  const { dy: dgy, dx: dgx } = gradient(gimage, w, h)
  // scikit-image always uses a 3x3 square here; the balloon's magnitude scales
  // the threshold rather than the structuring element.
  const ball: [number, number][] = []
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) ball.push([dy, dx])
  const balloonThreshold = balloon > 0 ? threshold / Math.abs(balloon) : threshold * Math.abs(balloon)

  for (let it = 0; it < iterations; it++) {
    if (balloon !== 0) {
      const moved = new Uint8Array(level.length)
      if (balloon > 0) dilateBy(level, w, h, ball, moved)
      else erodeBy(level, w, h, ball, moved)
      // The edge indicator is LARGE in flat regions and small on boundaries, so
      // the balloon pushes where there is no edge and stalls at one.
      for (let i = 0; i < level.length; i++) if (gimage[i] > balloonThreshold) level[i] = moved[i]
    }

    const asFloat = new Float32Array(level.length)
    for (let i = 0; i < level.length; i++) asFloat[i] = level[i]
    const { dy: duy, dx: dux } = gradient(asFloat, w, h)

    const next = new Uint8Array(level)
    for (let i = 0; i < level.length; i++) {
      const aux = dgy[i] * duy[i] + dgx[i] * dux[i]
      if (aux > 0) next[i] = 1
      else if (aux < 0) next[i] = 0
    }
    level = next
    for (let s = 0; s < smoothing; s++) level = curvatureStep(level, w, h, it + s)
  }
  return level
}

export const featActiveContour: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const source = (inputs.image ?? inputs.main) as any
  if (!source) return { main: null, mask: null, contour: [], count: 0 }

  const img = toBgr(cv, source)
  const w = img.cols
  const h = img.rows

  const hint = (message: string) => {
    const out = ctx.track(img.clone())
    cv.rectangle(out, new cv.Point(0, 0), new cv.Point(w, 30), new cv.Scalar(20, 20, 20, 255), -1)
    cv.putText(out, message, new cv.Point(8, 21), cv.FONT_HERSHEY_SIMPLEX, 0.55, new cv.Scalar(50, 200, 200, 255), 1, cv.LINE_AA)
    img.delete()
    return { main: out, mask: null, contour: [], count: 0 }
  }

  const state = (ctx.state.get(ctx.nodeId) as { lastRun: number; mask: Uint8Array | null } | undefined) ?? { lastRun: 0, mask: null }
  const runTrigger = Number(params.run) ? 1 : 0
  const fired = runTrigger === 1 && state.lastRun === 0
  state.lastRun = runTrigger
  ctx.state.set(ctx.nodeId, state)
  const live = !!params.live

  const render = (level: Uint8Array) => {
    const mask = ctx.track(new cv.Mat(h, w, cv.CV_8U))
    for (let i = 0; i < level.length; i++) mask.data[i] = level[i] ? 255 : 0

    const colour = hexToBgr(params.contour_color, [85, 51, 255])
    const opacity = (Number(params.fill_opacity) ?? 0) / 100
    const overlay = ctx.track(img.clone())
    if (opacity > 0) {
      const data = overlay.data
      for (let p = 0, i = 0; p < level.length; p++, i += 3) {
        if (!level[p]) continue
        data[i] = data[i] * (1 - opacity) + colour[0] * opacity
        data[i + 1] = data[i + 1] * (1 - opacity) + colour[1] * opacity
        data[i + 2] = data[i + 2] * (1 - opacity) + colour[2] * opacity
      }
    }

    const contours = new cv.MatVector()
    const hierarchy = new cv.Mat()
    cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    const thickness = Math.max(1, Math.round(Number(params.thickness) || 2))
    cv.drawContours(overlay, contours, -1, new cv.Scalar(colour[0], colour[1], colour[2], 255), thickness, cv.LINE_AA)

    const polygons: { _type: string; shape: string; pts: number[][]; relative: boolean }[] = []
    for (let c = 0; c < contours.size(); c++) {
      const mat = contours.get(c)
      const data = mat.data32S
      const pts: number[][] = []
      for (let i = 0; i < mat.rows; i++) pts.push([data[i * 2] / w, data[i * 2 + 1] / h])
      if (pts.length >= 3) polygons.push({ _type: 'graphics', shape: 'polygon', pts, relative: true })
    }
    contours.delete()
    hierarchy.delete()

    const count = cv.countNonZero(mask)
    img.delete()
    return { main: overlay, mask, contour: polygons, count }
  }

  if (!fired && !live) {
    if (state.mask) return render(state.mask)
    return hint('Press Run')
  }

  const gray = toGray(cv, img)
  const sigma = Number(params.pre_blur) ?? 2
  const smoothed = new cv.Mat()
  if (sigma > 0) {
    const k = Math.max(3, (Math.trunc(6 * sigma + 1) | 1))
    cv.GaussianBlur(gray, smoothed, new cv.Size(k, k), sigma, sigma, cv.BORDER_DEFAULT)
  } else gray.copyTo(smoothed)
  gray.delete()

  const image = new Float32Array(w * h)
  for (let i = 0; i < image.length; i++) image[i] = smoothed.data[i] / 255

  // Initial level set: a wired mask, a box around the seed points, or a
  // centred circle/ellipse sized by the Init Size parameter.
  const level = new Uint8Array(w * h)
  const maskIn = inputs.mask as any
  const { fg } = splitSeeds(inputs.points, w, h)
  if (maskIn) {
    let m = toGray(cv, maskIn)
    if (m.cols !== w || m.rows !== h) {
      const resized = new cv.Mat()
      cv.resize(m, resized, new cv.Size(w, h), 0, 0, cv.INTER_NEAREST)
      m.delete()
      m = resized
    }
    for (let i = 0; i < level.length; i++) level[i] = m.data[i] > 127 ? 1 : 0
    m.delete()
  } else {
    const pct = (Number(params.init_radius) ?? 45) / 100
    const shape = Math.round(Number(params.init_shape) || 0)
    const cx = fg.length ? Math.round(fg.reduce((s, p) => s + p[0], 0) / fg.length) : Math.round(w / 2)
    const cy = fg.length ? Math.round(fg.reduce((s, p) => s + p[1], 0) / fg.length) : Math.round(h / 2)
    const seed = new cv.Mat(h, w, cv.CV_8U, new cv.Scalar(0))
    if (shape === 2) seed.setTo(new cv.Scalar(1))
    else if (shape === 1) {
      cv.ellipse(seed, new cv.Point(cx, cy), new cv.Size(Math.round((w * pct) / 2), Math.round((h * pct) / 2)), 0, 0, 360, new cv.Scalar(1), -1)
    } else {
      cv.circle(seed, new cv.Point(cx, cy), Math.round((Math.min(w, h) * pct) / 2), new cv.Scalar(1), -1)
    }
    level.set(seed.data)
    seed.delete()
  }

  let sum = 0
  for (let i = 0; i < level.length; i++) sum += level[i]
  if (sum < 4) {
    smoothed.delete()
    return hint('Bad init region')
  }

  const iterations = Math.max(10, Math.round(Number(params.iterations) || 150))
  const smoothing = Math.max(0, Math.round(Number(params.smoothing) ?? 1))
  const method = Math.round(Number(params.method) ?? 1)

  let result: Uint8Array
  if (method === 2) {
    // Edge indicator: small where the gradient is strong, so the contour stalls
    // on boundaries. scikit-image's inverse_gaussian_gradient is
    // 1 / sqrt(1 + alpha * |∇I|) — the magnitude itself, not its square.
    const { dy, dx } = gradient(image, w, h)
    const g = new Float32Array(image.length)
    for (let i = 0; i < g.length; i++) g[i] = 1 / Math.sqrt(1 + 100 * Math.hypot(dy[i], dx[i]))
    const sorted = Float32Array.from(g).sort()
    const threshold = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.4))]
    result = morphGac(g, level, w, h, iterations, smoothing, Number(params.balloon) ?? -1, threshold)
  } else {
    // Classic Kass snakes need a parametric curve and a linear solve; the
    // morphological Chan-Vese below reaches the same region-based result with
    // a level set, so it also stands in for the "Classic Snake" option.
    const lambda = Number(params.cv_lambda) ?? 1
    result = morphChanVese(image, level, w, h, iterations, smoothing, lambda, 1)
  }
  smoothed.delete()

  // Chan-Vese has no notion of which side is the object, so it can settle on
  // the complement; keep whichever region agrees with the seed.
  let inside = 0
  let outside = 0
  let filled = 0
  for (let i = 0; i < result.length; i++) {
    if (!result[i]) continue
    filled++
    if (level[i]) inside++
    else outside++
  }
  if (outside > inside && filled > result.length * 0.5) {
    for (let i = 0; i < result.length; i++) result[i] = result[i] ? 0 : 1
  }

  state.mask = result
  return render(result)
}

/* ------------------------------------------------------- RANSAC homography */

/** Brute-force match with Lowe's ratio test; returns the surviving pairs. */
function matchDescriptors(cv: any, des1: any, des2: any, norm: number, ratio: number): { q: number; t: number }[] {
  const matcher = new cv.BFMatcher(norm, false)
  const knn = new cv.DMatchVectorVector()
  const good: { q: number; t: number }[] = []
  try {
    matcher.knnMatch(des1, des2, knn, 2)
    for (let i = 0; i < knn.size(); i++) {
      const pair = knn.get(i)
      if (pair.size() === 2) {
        const best = pair.get(0)
        const second = pair.get(1)
        if (best.distance < ratio * second.distance) good.push({ q: best.queryIdx, t: best.trainIdx })
      }
      pair.delete()
    }
  } catch {
    // Mismatched descriptor types, e.g. an L2 norm against binary descriptors.
  }
  knn.delete()
  matcher.delete()
  return good
}

function keypointPixels(list: unknown, w: number, h: number): [number, number][] {
  return (Array.isArray(list) ? list : []).map((entry) => {
    const p = (entry as { pts?: number[][] })?.pts?.[0] ?? [0, 0]
    return [p[0] * w, p[1] * h] as [number, number]
  })
}

/**
 * Similarity transform (rotation, uniform scale, translation) from two point
 * pairs, wrapped in RANSAC. `cv.estimateAffinePartial2D` is not in this build.
 */
function estimatePartialAffine(src: [number, number][], dst: [number, number][], threshold: number, robust: boolean): { matrix: number[]; inliers: boolean[] } | null {
  const n = src.length
  if (n < 2) return null

  const fit = (indices: number[]): number[] | null => {
    // Least-squares similarity fit over the given correspondences.
    let sx = 0, sy = 0, dx = 0, dy = 0
    for (const i of indices) {
      sx += src[i][0]
      sy += src[i][1]
      dx += dst[i][0]
      dy += dst[i][1]
    }
    const m = indices.length
    sx /= m; sy /= m; dx /= m; dy /= m
    let a = 0, b = 0, varSrc = 0
    for (const i of indices) {
      const ux = src[i][0] - sx
      const uy = src[i][1] - sy
      const vx = dst[i][0] - dx
      const vy = dst[i][1] - dy
      a += ux * vx + uy * vy
      b += ux * vy - uy * vx
      varSrc += ux * ux + uy * uy
    }
    if (varSrc < 1e-12) return null
    const cosScale = a / varSrc
    const sinScale = b / varSrc
    return [cosScale, -sinScale, dx - (cosScale * sx - sinScale * sy), sinScale, cosScale, dy - (sinScale * sx + cosScale * sy)]
  }

  const residual = (m: number[], i: number) => {
    const x = m[0] * src[i][0] + m[1] * src[i][1] + m[2]
    const y = m[3] * src[i][0] + m[4] * src[i][1] + m[5]
    return Math.hypot(x - dst[i][0], y - dst[i][1])
  }

  if (!robust) {
    const all = src.map((_, i) => i)
    const m = fit(all)
    return m ? { matrix: m, inliers: all.map(() => true) } : null
  }

  let bestCount = -1
  let bestInliers: boolean[] | null = null
  for (let iter = 0; iter < 500; iter++) {
    const i = Math.floor(Math.random() * n)
    const j = Math.floor(Math.random() * n)
    if (i === j) continue
    const candidate = fit([i, j])
    if (!candidate) continue
    const inliers: boolean[] = []
    let count = 0
    for (let k = 0; k < n; k++) {
      const ok = residual(candidate, k) < threshold
      inliers.push(ok)
      if (ok) count++
    }
    if (count > bestCount) {
      bestCount = count
      bestInliers = inliers
    }
  }
  if (!bestInliers || bestCount < 2) return null
  const refined = fit(bestInliers.map((ok, i) => (ok ? i : -1)).filter((i) => i >= 0))
  return refined ? { matrix: refined, inliers: bestInliers } : null
}

export const cvRansac: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const img1 = inputs.img1 as any
  const img2 = inputs.img2 as any
  const empty = { warped: img1 ?? null, overlay: img1 ?? null, homography: null, inliers: 0 }
  if (!img1 || !img2) return empty

  const kp1 = inputs.kp1
  const kp2 = inputs.kp2
  const des1 = inputs.des1 as any
  const des2 = inputs.des2 as any
  if (!Array.isArray(kp1) || !Array.isArray(kp2) || !kp1.length || !kp2.length || !des1 || !des2) return empty

  const norm = Math.round(Number(params.norm) || 0) === 0 ? cv.NORM_L2 : cv.NORM_HAMMING
  const ratio = Number(params.ratio) || 0.75
  const threshold = Number(params.ransac_thresh) || 5
  const minInliers = Math.max(4, Math.round(Number(params.min_inliers) || 10))
  const model = Math.round(Number(params.model) || 0)
  const ordinary = Math.round(Number(params.method) || 0) === 1

  const good = matchDescriptors(cv, des1, des2, norm, ratio)
  if (good.length < Math.max(4, minInliers)) return { ...empty, inliers: good.length }

  const px1 = keypointPixels(kp1, img1.cols, img1.rows)
  const px2 = keypointPixels(kp2, img2.cols, img2.rows)
  const src = good.map((m) => px1[m.q]).filter(Boolean) as [number, number][]
  const dst = good.map((m) => px2[m.t]).filter(Boolean) as [number, number][]
  if (src.length < 4 || src.length !== dst.length) return { ...empty, inliers: 0 }

  let matrix: number[] | null = null
  let inlierFlags: boolean[] = []

  if (model === 0) {
    const srcMat = cv.matFromArray(src.length, 1, cv.CV_32FC2, src.flat())
    const dstMat = cv.matFromArray(dst.length, 1, cv.CV_32FC2, dst.flat())
    const inlierMask = new cv.Mat()
    // "Ordinary" skips outlier rejection entirely, which is the point of the
    // node: one bad correspondence is enough to wreck a non-robust fit.
    const H = ordinary ? cv.findHomography(srcMat, dstMat, 0) : cv.findHomography(srcMat, dstMat, cv.RANSAC, threshold, inlierMask)
    if (H && !H.empty()) {
      matrix = Array.from(H.data64F as Float64Array)
      inlierFlags = ordinary ? src.map(() => true) : Array.from(inlierMask.data as Uint8Array, (v) => v !== 0)
    }
    H?.delete?.()
    inlierMask.delete()
    srcMat.delete()
    dstMat.delete()
  } else {
    const fit = estimatePartialAffine(src, dst, threshold, !ordinary)
    if (fit) {
      matrix = [...fit.matrix, 0, 0, 1]
      inlierFlags = fit.inliers
    }
  }

  if (!matrix) return { ...empty, inliers: 0 }
  const inlierCount = inlierFlags.filter(Boolean).length

  const homography = cv.matFromArray(3, 3, cv.CV_64F, matrix)
  const warped = ctx.track(new cv.Mat())
  cv.warpPerspective(toBgr(cv, img1), warped, homography, new cv.Size(img2.cols, img2.rows))
  homography.delete()

  // cv.KeyPoint is not constructible here, so drawMatches is unreachable and
  // the inlier lines are drawn onto a side-by-side composite instead.
  const a = toBgr(cv, img1)
  const b = toBgr(cv, img2)
  const overlay = ctx.track(new cv.Mat(Math.max(a.rows, b.rows), a.cols + b.cols, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 255)))
  a.copyTo(overlay.roi(new cv.Rect(0, 0, a.cols, a.rows)))
  b.copyTo(overlay.roi(new cv.Rect(a.cols, 0, b.cols, b.rows)))

  const green = new cv.Scalar(0, 220, 80, 255)
  const maxDisplay = Math.max(1, Math.round(Number(params.max_display) || 60))
  let drawn = 0
  for (let i = 0; i < good.length && drawn < maxDisplay; i++) {
    if (!inlierFlags[i]) continue
    const from = px1[good[i].q]
    const to = px2[good[i].t]
    if (!from || !to) continue
    cv.line(overlay, new cv.Point(Math.round(from[0]), Math.round(from[1])), new cv.Point(Math.round(to[0]) + a.cols, Math.round(to[1])), green, 1, cv.LINE_AA)
    drawn++
  }
  cv.putText(overlay, `Inliers: ${inlierCount}/${good.length}`, new cv.Point(8, 22), cv.FONT_HERSHEY_SIMPLEX, 0.65, green, 1, cv.LINE_AA)
  a.delete()
  b.delete()

  ctx.emit('inliers', inlierCount)
  return { warped, overlay, homography: [matrix.slice(0, 3), matrix.slice(3, 6), matrix.slice(6, 9)], inliers: inlierCount }
}

/* ------------------------------------------------- Monte-Carlo clustering */

const MC_COLOURS = [infernoColor, viridisColor, jetColor, oceanColor]

/** Lloyd's k-means over an index subset, returning the centroids. */
function kmeansCentroids(features: Float32Array, channels: number, indices: Int32Array, k: number, random: () => number): Float32Array {
  const centroids = new Float32Array(k * channels)
  // k-means++ style seeding: spread the initial centres out.
  const first = indices[Math.floor(random() * indices.length)]
  for (let c = 0; c < channels; c++) centroids[c] = features[first * channels + c]
  const distances = new Float32Array(indices.length).fill(Infinity)
  for (let centre = 1; centre < k; centre++) {
    let total = 0
    for (let i = 0; i < indices.length; i++) {
      let d = 0
      for (let c = 0; c < channels; c++) {
        const diff = features[indices[i] * channels + c] - centroids[(centre - 1) * channels + c]
        d += diff * diff
      }
      if (d < distances[i]) distances[i] = d
      total += distances[i]
    }
    let target = random() * total
    let pick = indices[indices.length - 1]
    for (let i = 0; i < indices.length; i++) {
      target -= distances[i]
      if (target <= 0) {
        pick = indices[i]
        break
      }
    }
    for (let c = 0; c < channels; c++) centroids[centre * channels + c] = features[pick * channels + c]
  }

  const sums = new Float64Array(k * channels)
  const counts = new Int32Array(k)
  for (let iter = 0; iter < 50; iter++) {
    sums.fill(0)
    counts.fill(0)
    for (let i = 0; i < indices.length; i++) {
      const base = indices[i] * channels
      let best = 0
      let bestDistance = Infinity
      for (let centre = 0; centre < k; centre++) {
        let d = 0
        for (let c = 0; c < channels; c++) {
          const diff = features[base + c] - centroids[centre * channels + c]
          d += diff * diff
        }
        if (d < bestDistance) {
          bestDistance = d
          best = centre
        }
      }
      counts[best]++
      for (let c = 0; c < channels; c++) sums[best * channels + c] += features[base + c]
    }
    let moved = 0
    for (let centre = 0; centre < k; centre++) {
      if (!counts[centre]) continue
      for (let c = 0; c < channels; c++) {
        const next = sums[centre * channels + c] / counts[centre]
        moved += Math.abs(next - centroids[centre * channels + c])
        centroids[centre * channels + c] = next
      }
    }
    if (moved < 1e-4) break
  }
  return centroids
}

export const cvMontecarloCluster: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = (inputs.image ?? inputs.main) as any
  if (!image) return { main: null, probability: null, prob_raw: null, stats: {} }

  const w = image.cols
  const h = image.rows
  const channels = image.channels()
  const n = w * h

  // Each channel is stretched to 0–1 so no band dominates the distance.
  const features = new Float32Array(n * channels)
  const data = image.data
  const mins = new Float32Array(channels).fill(Infinity)
  const maxs = new Float32Array(channels).fill(-Infinity)
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < channels; c++) {
      const v = data[i * channels + c]
      features[i * channels + c] = v
      if (v < mins[c]) mins[c] = v
      if (v > maxs[c]) maxs[c] = v
    }
  }
  for (let c = 0; c < channels; c++) {
    const range = maxs[c] - mins[c] || 1
    for (let i = 0; i < n; i++) features[i * channels + c] = (features[i * channels + c] - mins[c]) / range
  }

  let pool: Int32Array
  const roi = inputs.roi as any
  if (roi) {
    let m = toGray(cv, roi)
    if (m.cols !== w || m.rows !== h) {
      const resized = new cv.Mat()
      cv.resize(m, resized, new cv.Size(w, h), 0, 0, cv.INTER_NEAREST)
      m.delete()
      m = resized
    }
    const kept: number[] = []
    for (let i = 0; i < n; i++) if (m.data[i] > 0) kept.push(i)
    m.delete()
    pool = Int32Array.from(kept)
  } else {
    pool = new Int32Array(n)
    for (let i = 0; i < n; i++) pool[i] = i
  }
  if (pool.length < 4) return { main: image, probability: null, prob_raw: null, stats: { error: 'ROI empty' } }

  const iterations = Math.max(5, Math.round(Number(params.n_iterations) || 40))
  const fraction = (Number(params.subsample) ?? 30) / 100
  const k = Math.max(2, Math.round(Number(params.n_clusters) || 2))
  const refChannel = Math.min(Math.max(0, Math.round(Number(params.ref_channel) || 0)), channels - 1)
  const anchorHigh = Math.round(Number(params.anchor_rule) || 0) === 0

  let seed = (Math.round(Number(params.seed) || 0) >>> 0) || 1
  const random = () => {
    seed = (seed + 0x6d2b79f5) >>> 0
    let t = seed
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }

  const sampleCount = Math.max(k, Math.floor(pool.length * fraction))
  const votes = new Float32Array(n)

  // Each round clusters a fresh subsample and votes for the anchor cluster.
  // The share of rounds a pixel wins is its membership probability, which is
  // exactly the uncertainty a single k-means run hides.
  for (let round = 0; round < iterations; round++) {
    const take = Math.min(sampleCount, pool.length)
    const shuffled = Int32Array.from(pool)
    for (let i = 0; i < take; i++) {
      const j = i + Math.floor(random() * (shuffled.length - i))
      const tmp = shuffled[i]
      shuffled[i] = shuffled[j]
      shuffled[j] = tmp
    }
    const subset = shuffled.subarray(0, take)
    const centroids = kmeansCentroids(features, channels, subset, k, random)

    let target = 0
    let bestValue = anchorHigh ? -Infinity : Infinity
    for (let centre = 0; centre < k; centre++) {
      const value = centroids[centre * channels + refChannel]
      if (anchorHigh ? value > bestValue : value < bestValue) {
        bestValue = value
        target = centre
      }
    }

    for (let i = 0; i < n; i++) {
      let best = 0
      let bestDistance = Infinity
      for (let centre = 0; centre < k; centre++) {
        let d = 0
        for (let c = 0; c < channels; c++) {
          const diff = features[i * channels + c] - centroids[centre * channels + c]
          d += diff * diff
        }
        if (d < bestDistance) {
          bestDistance = d
          best = centre
        }
      }
      if (best === target) votes[i] += 1
    }
  }

  const probability = ctx.track(new cv.Mat(h, w, cv.CV_32F))
  const probabilityData = probability.data32F
  const eight = new cv.Mat(h, w, cv.CV_8U)
  let sum = 0
  for (let i = 0; i < n; i++) {
    const p = votes[i] / iterations
    probabilityData[i] = p
    eight.data[i] = Math.round(Math.max(0, Math.min(1, p)) * 255)
    sum += p
  }

  const heat = ctx.track(applyColormap(cv, eight, MC_COLOURS[Math.min(MC_COLOURS.length - 1, Math.round(Number(params.colormap) || 0))]))
  eight.delete()

  let uncertain = 0
  for (let i = 0; i < n; i++) {
    const p = probabilityData[i]
    if (p > 0.2 && p < 0.8) uncertain++
  }

  return {
    main: heat,
    probability: heat,
    prob_raw: probability,
    stats: {
      iterations,
      clusters: k,
      mean_probability: Math.round((sum / n) * 1e4) / 1e4,
      // Pixels the rounds disagreed about: the honest measure of how stable
      // the segmentation actually is.
      uncertain_pct: Math.round((uncertain / n) * 1000) / 10,
    },
  }
}

/** Internals exposed for the verification harness, not for node code. */
export const __testing = { supInf, infSup, curvatureStep, morphChanVese, morphGac, gradient }
