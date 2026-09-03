import type { NodeImpl } from '../types'
import { toBgr, toGray } from '../cvUtils'

const COLOR_INT_W = 160
const COLOR_INT_H = 120
const MOTION_INT_W = 320
const MOTION_INT_H = 240

function boundingBoxOf(cv: any, ctx: any, maskGray: any, w: number, h: number, pad: number): [number, number, number, number] {
  const rect = cv.boundingRect(maskGray)
  const x1 = Math.max(0, rect.x - pad)
  const y1 = Math.max(0, rect.y - pad)
  const x2 = Math.min(w, rect.x + rect.width + pad)
  const y2 = Math.min(h, rect.y + rect.height + pad)
  return [x1, y1, x2, y2]
}

function maskAny(data: Uint8Array): boolean {
  for (let i = 0; i < data.length; i++) if (data[i] > 0) return true
  return false
}

// ---------------------------------------------------------------------------
// EVM Color
// ---------------------------------------------------------------------------
interface EvmColorState {
  low1: Float32Array | null
  low2: Float32Array | null
  shape: [number, number, number] | null
  sig: string
}

export const pluginEvmColor: NodeImpl = (inputs, params, ctx) => {
  let state: EvmColorState = ctx.state.get(ctx.nodeId)
  if (!state) {
    state = { low1: null, low2: null, shape: null, sig: '' }
    ctx.state.set(ctx.nodeId, state)
  }
  const cv = ctx.cv
  const img = inputs.image as any
  if (!img) return { main: null, signal: 0, signal_cb: 0, filtered_vis: null }

  const alpha = Number(params.alpha) || 50
  const lowHz = (Number(params.low_cutoff) || 830) / 1000
  const highHz = (Number(params.high_cutoff) || 1000) / 1000
  const fps = Math.max(1, Number(params.fps) || 30)
  const levels = Math.round(Number(params.levels) || 4)
  const att = (Number(params.attenuation) || 3) / 100

  const sig = `${lowHz}:${highHz}:${fps}:${levels}`
  if (sig !== state.sig) {
    state.low1 = null
    state.low2 = null
    state.sig = sig
  }

  const rHigh = 1 - Math.exp((-2 * Math.PI * highHz) / fps)
  const rLow = 1 - Math.exp((-2 * Math.PI * lowHz) / fps)

  const bgr = ctx.track(toBgr(cv, img))
  const w = bgr.cols
  const h = bgr.rows

  const maskIn = inputs.mask as any
  let hasMask = false
  let x1 = 0, y1 = 0, x2 = w, y2 = h
  let maskGray: any = null
  if (maskIn) {
    let mg = ctx.track(toGray(cv, maskIn))
    if (mg.cols !== w || mg.rows !== h) {
      const resized = ctx.track(new cv.Mat())
      cv.resize(mg, resized, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR)
      mg = resized
    }
    if (maskAny(mg.data as Uint8Array)) {
      hasMask = true
      maskGray = mg
      ;[x1, y1, x2, y2] = boundingBoxOf(cv, ctx, mg, w, h, 8)
    }
  }

  const roi = ctx.track(bgr.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1)))
  const small = ctx.track(new cv.Mat())
  cv.resize(roi, small, new cv.Size(COLOR_INT_W, COLOR_INT_H), 0, 0, cv.INTER_LINEAR)
  const smallF = ctx.track(new cv.Mat())
  small.convertTo(smallF, cv.CV_32FC3, 1 / 255)
  const ycrcbS = ctx.track(new cv.Mat())
  cv.cvtColor(smallF, ycrcbS, cv.COLOR_BGR2YCrCb)

  let coarse = ycrcbS
  for (let i = 0; i < levels; i++) {
    const down = ctx.track(new cv.Mat())
    cv.pyrDown(coarse, down)
    coarse = down
  }
  const coarseData = coarse.data32F as Float32Array
  const cw = coarse.cols
  const ch = coarse.rows
  const n = cw * ch * 3

  if (!state.low1) {
    state.low1 = Float32Array.from(coarseData)
    state.low2 = Float32Array.from(coarseData)
  }
  const filtered = new Float32Array(n)
  for (let i = 0; i < n; i++) {
    state.low1![i] = (1 - rHigh) * state.low1![i] + rHigh * coarseData[i]
    state.low2![i] = (1 - rLow) * state.low2![i] + rLow * coarseData[i]
    filtered[i] = state.low1![i] - state.low2![i]
  }

  // Signal: mean of Cr/Cb channels, restricted to masked coarse pixels if present.
  let signal = 0, signalCb = 0, count = 0
  if (hasMask && maskGray) {
    const maskSmall = ctx.track(new cv.Mat())
    const maskRoi = ctx.track(maskGray.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1)))
    cv.resize(maskRoi, maskSmall, new cv.Size(cw, ch), 0, 0, cv.INTER_LINEAR)
    const maskSmallData = maskSmall.data as Uint8Array
    for (let i = 0; i < cw * ch; i++) {
      if (maskSmallData[i] > 0) {
        signal += filtered[i * 3 + 1]
        signalCb += filtered[i * 3 + 2]
        count++
      }
    }
  }
  if (count === 0) {
    count = cw * ch
    signal = 0
    signalCb = 0
    for (let i = 0; i < count; i++) {
      signal += filtered[i * 3 + 1]
      signalCb += filtered[i * 3 + 2]
    }
  }
  signal = (signal / count) * 1000
  signalCb = (signalCb / count) * 1000

  const amplified = ctx.track(new cv.Mat(ch, cw, cv.CV_32FC3))
  const ampData = amplified.data32F as Float32Array
  for (let i = 0; i < n; i++) ampData[i] = Math.max(-att, Math.min(att, filtered[i])) * alpha

  let amp: any = amplified
  for (let i = 0; i < levels; i++) {
    const up = ctx.track(new cv.Mat())
    cv.pyrUp(amp, up)
    amp = up
  }
  let ampFinal = amp
  if (amp.cols !== COLOR_INT_W || amp.rows !== COLOR_INT_H) {
    const cropped = ctx.track(new cv.Mat())
    cv.resize(amp, cropped, new cv.Size(COLOR_INT_W, COLOR_INT_H), 0, 0, cv.INTER_LINEAR)
    ampFinal = cropped
  }

  const roiW = x2 - x1
  const roiH = y2 - y1
  const ampRoi = ctx.track(new cv.Mat())
  cv.resize(ampFinal, ampRoi, new cv.Size(roiW, roiH), 0, 0, cv.INTER_LINEAR)

  const roiF = ctx.track(new cv.Mat())
  roi.convertTo(roiF, cv.CV_32FC3, 1 / 255)
  const roiYcrcb = ctx.track(new cv.Mat())
  cv.cvtColor(roiF, roiYcrcb, cv.COLOR_BGR2YCrCb)
  const roiYcrcbData = roiYcrcb.data32F as Float32Array
  const ampRoiData = ampRoi.data32F as Float32Array
  let maskRoiFull: Uint8Array | null = null
  if (hasMask && maskGray) {
    const mr = ctx.track(maskGray.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1)))
    maskRoiFull = mr.data as Uint8Array
  }
  for (let i = 0, px = 0; i < roiW * roiH; i++, px += 3) {
    const weight = maskRoiFull ? (maskRoiFull[i] > 0 ? 1 : 0) : 1
    roiYcrcbData[px + 1] = Math.max(0, Math.min(1, roiYcrcbData[px + 1] + ampRoiData[px + 1] * weight))
    roiYcrcbData[px + 2] = Math.max(0, Math.min(1, roiYcrcbData[px + 2] + ampRoiData[px + 2] * weight))
  }
  const roiResult = ctx.track(new cv.Mat())
  cv.cvtColor(roiYcrcb, roiResult, cv.COLOR_YCrCb2BGR)

  const result = ctx.track(bgr.clone())
  const resultF = ctx.track(new cv.Mat())
  result.convertTo(resultF, cv.CV_32FC3, 1 / 255)
  const roiResultByte = ctx.track(new cv.Mat())
  roiResult.convertTo(roiResultByte, cv.CV_8UC3, 255)
  const target = result.roi(new cv.Rect(x1, y1, roiW, roiH))
  roiResultByte.copyTo(target)
  target.delete()

  // Delta visualization.
  const ampFullVis = ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(0, 0, 0)))
  const visData = ampFullVis.data as Uint8Array
  for (let i = 0, px = 0; i < roiW * roiH; i++, px += 3) {
    const gy = y1 + Math.floor(i / roiW)
    const gx = x1 + (i % roiW)
    const outPx = (gy * w + gx) * 3
    const r = Math.round(Math.max(0, Math.min(1, 0.5 + ampRoiData[px + 2] * 2)) * 255)
    const b = Math.round(Math.max(0, Math.min(1, 0.5 + ampRoiData[px + 1] * 2)) * 255)
    visData[outPx] = b
    visData[outPx + 2] = r
  }

  return { main: result, signal, signal_cb: signalCb, filtered_vis: ampFullVis }
}

// ---------------------------------------------------------------------------
// EVM Motion
// ---------------------------------------------------------------------------
interface EvmMotionState {
  low1: any[] | null
  low2: any[] | null
  sig: string
}

function buildLaplacianPyramid(cv: any, ctx: any, img: any, levels: number): any[] {
  const pyr: any[] = []
  let current = img
  for (let i = 0; i < levels; i++) {
    const down = ctx.track(new cv.Mat())
    cv.pyrDown(current, down)
    const up = ctx.track(new cv.Mat())
    cv.pyrUp(down, up, new cv.Size(current.cols, current.rows))
    const lap = ctx.track(new cv.Mat())
    cv.subtract(current, up, lap)
    pyr.push(lap)
    current = down
  }
  pyr.push(current)
  return pyr
}

function collapseLaplacianPyramid(cv: any, ctx: any, pyr: any[]): any {
  let result = pyr[pyr.length - 1]
  for (let i = pyr.length - 2; i >= 0; i--) {
    const lap = pyr[i]
    const up = ctx.track(new cv.Mat())
    cv.pyrUp(result, up, new cv.Size(lap.cols, lap.rows))
    const summed = ctx.track(new cv.Mat())
    cv.add(up, lap, summed)
    result = summed
  }
  return result
}

export const pluginEvmMotion: NodeImpl = (inputs, params, ctx) => {
  let state: EvmMotionState = ctx.state.get(ctx.nodeId)
  if (!state) {
    state = { low1: null, low2: null, sig: '' }
    ctx.state.set(ctx.nodeId, state)
  }
  const cv = ctx.cv
  const img = inputs.image as any
  if (!img) return { main: null, motion_mag: 0, motion_vis: null }

  const alpha = Number(params.alpha) || 30
  const lowHz = (Number(params.low_cutoff) || 400) / 1000
  const highHz = (Number(params.high_cutoff) || 3000) / 1000
  const fps = Math.max(1, Number(params.fps) || 30)
  const levels = Math.round(Number(params.levels) || 6)
  const lambdaC = Math.max(1, Number(params.lambda_c) || 16)
  const att = (Number(params.attenuation) || 10) / 100

  const sig = `${lowHz}:${highHz}:${fps}:${levels}:${lambdaC}`
  if (sig !== state.sig) {
    state.low1 = null
    state.low2 = null
    state.sig = sig
  }

  const rHigh = 1 - Math.exp((-2 * Math.PI * highHz) / fps)
  const rLow = 1 - Math.exp((-2 * Math.PI * lowHz) / fps)

  const bgr = ctx.track(toBgr(cv, img))
  const w = bgr.cols
  const h = bgr.rows

  const maskIn = inputs.mask as any
  let hasMask = false
  let x1 = 0, y1 = 0, x2 = w, y2 = h
  let maskGray: any = null
  if (maskIn) {
    let mg = ctx.track(toGray(cv, maskIn))
    if (mg.cols !== w || mg.rows !== h) {
      const resized = ctx.track(new cv.Mat())
      cv.resize(mg, resized, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR)
      mg = resized
    }
    if (maskAny(mg.data as Uint8Array)) {
      hasMask = true
      maskGray = mg
      ;[x1, y1, x2, y2] = boundingBoxOf(cv, ctx, mg, w, h, 8)
    }
  }

  const roi = ctx.track(bgr.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1)))
  const frameSmallByte = ctx.track(new cv.Mat())
  cv.resize(roi, frameSmallByte, new cv.Size(MOTION_INT_W, MOTION_INT_H), 0, 0, cv.INTER_LINEAR)
  const frameSmall = ctx.track(new cv.Mat())
  frameSmallByte.convertTo(frameSmall, cv.CV_32FC3, 1 / 255)

  const pyr = buildLaplacianPyramid(cv, ctx, frameSmall, levels)

  if (!state.low1) {
    state.low1 = pyr.map((lv) => lv.clone())
    state.low2 = pyr.map((lv) => lv.clone())
  }

  const ampPyr: any[] = []
  let motionSignal = 0

  for (let l = 0; l < pyr.length; l++) {
    const lv = pyr[l]
    const wavelengthL = 2 ** (l + 1)
    const alphaEff = alpha * Math.min(1, wavelengthL / lambdaC)

    const low1Data = state.low1![l].data32F as Float32Array
    const low2Data = state.low2![l].data32F as Float32Array
    const lvData = lv.data32F as Float32Array
    const n = lvData.length
    const filtered = new Float32Array(n)
    for (let i = 0; i < n; i++) {
      low1Data[i] = (1 - rHigh) * low1Data[i] + rHigh * lvData[i]
      low2Data[i] = (1 - rLow) * low2Data[i] + rLow * lvData[i]
      filtered[i] = Math.max(-att, Math.min(att, low1Data[i] - low2Data[i]))
    }

    let count = 0
    let sum = 0
    if (hasMask && maskGray) {
      const lw = lv.cols
      const lh = lv.rows
      const maskRoi = ctx.track(maskGray.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1)))
      const maskLv = ctx.track(new cv.Mat())
      cv.resize(maskRoi, maskLv, new cv.Size(lw, lh), 0, 0, cv.INTER_LINEAR)
      const maskLvData = maskLv.data as Uint8Array
      for (let i = 0; i < lw * lh; i++) {
        if (maskLvData[i] > 0) {
          sum += Math.abs(filtered[i * 3]) + Math.abs(filtered[i * 3 + 1]) + Math.abs(filtered[i * 3 + 2])
          count += 3
        }
      }
    }
    if (count === 0) {
      count = n
      sum = 0
      for (let i = 0; i < n; i++) sum += Math.abs(filtered[i])
    }
    motionSignal += (sum / count) * alphaEff

    const amped = ctx.track(new cv.Mat(lv.rows, lv.cols, cv.CV_32FC3))
    const ampedData = amped.data32F as Float32Array
    for (let i = 0; i < n; i++) ampedData[i] = lvData[i] + alphaEff * filtered[i]
    ampPyr.push(amped)
  }

  const reconstructedSmall = collapseLaplacianPyramid(cv, ctx, ampPyr)
  const reconstructedData = reconstructedSmall.data32F as Float32Array
  for (let i = 0; i < reconstructedData.length; i++) reconstructedData[i] = Math.max(0, Math.min(1, reconstructedData[i]))

  const roiW = x2 - x1
  const roiH = y2 - y1
  const reconstructedRoi = ctx.track(new cv.Mat())
  cv.resize(reconstructedSmall, reconstructedRoi, new cv.Size(roiW, roiH), 0, 0, cv.INTER_LINEAR)

  const result = ctx.track(bgr.clone())
  const reconByte = ctx.track(new cv.Mat())
  reconstructedRoi.convertTo(reconByte, cv.CV_8UC3, 255)

  if (hasMask && maskGray) {
    const maskRoi = ctx.track(maskGray.roi(new cv.Rect(x1, y1, x2 - x1, y2 - y1)))
    const maskData = maskRoi.data as Uint8Array
    const origRoi = ctx.track(result.roi(new cv.Rect(x1, y1, roiW, roiH)).clone())
    const origData = origRoi.data as Uint8Array
    const reconData = reconByte.data as Uint8Array
    const blended = ctx.track(new cv.Mat(roiH, roiW, cv.CV_8UC3))
    const blendedData = blended.data as Uint8Array
    for (let i = 0, px = 0; i < roiW * roiH; i++, px += 3) {
      const weight = maskData[i] > 0 ? 1 : 0
      blendedData[px] = weight ? reconData[px] : origData[px]
      blendedData[px + 1] = weight ? reconData[px + 1] : origData[px + 1]
      blendedData[px + 2] = weight ? reconData[px + 2] : origData[px + 2]
    }
    const target = result.roi(new cv.Rect(x1, y1, roiW, roiH))
    blended.copyTo(target)
    target.delete()
  } else {
    const target = result.roi(new cv.Rect(x1, y1, roiW, roiH))
    reconByte.copyTo(target)
    target.delete()
  }

  // Motion delta visualization.
  const origSmall = ctx.track(new cv.Mat())
  cv.resize(frameSmall, origSmall, new cv.Size(MOTION_INT_W, MOTION_INT_H), 0, 0, cv.INTER_LINEAR)
  const origSmallData = origSmall.data32F as Float32Array
  const deltaSmall = ctx.track(new cv.Mat(MOTION_INT_H, MOTION_INT_W, cv.CV_32FC3))
  const deltaData = deltaSmall.data32F as Float32Array
  const reconstructedSmallResized = ctx.track(new cv.Mat())
  cv.resize(reconstructedSmall, reconstructedSmallResized, new cv.Size(MOTION_INT_W, MOTION_INT_H), 0, 0, cv.INTER_LINEAR)
  const reconSmallData = reconstructedSmallResized.data32F as Float32Array
  for (let i = 0; i < deltaData.length; i++) deltaData[i] = Math.max(0, Math.min(1, 0.5 + (reconSmallData[i] - origSmallData[i]) * 3))

  const motionVisRoi = ctx.track(new cv.Mat())
  cv.resize(deltaSmall, motionVisRoi, new cv.Size(roiW, roiH), 0, 0, cv.INTER_LINEAR)
  const motionVisRoiByte = ctx.track(new cv.Mat())
  motionVisRoi.convertTo(motionVisRoiByte, cv.CV_8UC3, 255)

  const motionVis = ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(128, 128, 128)))
  const motionVisTarget = motionVis.roi(new cv.Rect(x1, y1, roiW, roiH))
  motionVisRoiByte.copyTo(motionVisTarget)
  motionVisTarget.delete()

  return { main: result, motion_mag: motionSignal * 1000, motion_vis: motionVis }
}
