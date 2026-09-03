import type { NodeImpl } from '../types'
import { toBgr } from '../cvUtils'

function applyLut(cv: any, src: any, lut: Uint8Array): any {
  const out = new cv.Mat()
  const channels = new cv.MatVector()
  cv.split(src, channels)
  for (let i = 0; i < channels.size(); i++) {
    const ch = channels.get(i)
    const data = ch.data as Uint8Array
    for (let j = 0; j < data.length; j++) data[j] = lut[data[j]]
    channels.set(i, ch)
  }
  cv.merge(channels, out)
  channels.delete()
  return out
}

export const cvGamma: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv
  const gamma = Math.max(0.01, Number(params.gamma) || 2.2)
  const linearise = params.linearise !== false
  const exponent = linearise ? gamma : 1 / gamma

  const lut = new Uint8Array(256)
  for (let i = 0; i < 256; i++) lut[i] = Math.round(Math.pow(i / 255, exponent) * 255)

  const bgr = ctx.track(toBgr(cv, src))
  return { main: ctx.track(applyLut(cv, bgr, lut)) }
}

export const cvLevels: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv

  const inBlack = Number(params.in_black) || 0
  const inWhite = params.in_white === undefined ? 255 : Number(params.in_white)
  const gamma = Number(params.gamma) || 1.0
  const outBlack = Number(params.out_black) || 0
  const outWhite = params.out_white === undefined ? 255 : Number(params.out_white)

  const diff = inWhite - inBlack || 0.001
  const lut = new Uint8Array(256)
  for (let i = 0; i < 256; i++) {
    let v = (i - inBlack) / diff
    v = Math.max(0, Math.min(1, v))
    if (gamma !== 1.0) v = Math.pow(v, 1.0 / gamma)
    v = v * (outWhite - outBlack) + outBlack
    lut[i] = Math.max(0, Math.min(255, Math.round(v)))
  }

  const bgr = ctx.track(toBgr(cv, src))
  return { main: ctx.track(applyLut(cv, bgr, lut)) }
}

export const cvShadowHighlight: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv

  const sAmt = (Number(params.shadows) || 0) / 100
  const hAmt = (Number(params.highlights) || 0) / 100
  let radius = Math.round(Number(params.radius) || 41)
  if (radius % 2 === 0) radius += 1

  const bgr = ctx.track(toBgr(cv, src))
  const lab = ctx.track(new cv.Mat())
  cv.cvtColor(bgr, lab, cv.COLOR_BGR2Lab)

  const channels = new cv.MatVector()
  cv.split(lab, channels)
  const l = channels.get(0)
  const lFloat = ctx.track(new cv.Mat())
  l.convertTo(lFloat, cv.CV_32F)

  const white = ctx.track(new cv.Mat(lFloat.rows, lFloat.cols, cv.CV_32F, new cv.Scalar(255)))
  const shadowMask = ctx.track(new cv.Mat())
  cv.subtract(white, lFloat, shadowMask)
  const shadowBlurred = ctx.track(new cv.Mat())
  cv.GaussianBlur(shadowMask, shadowBlurred, new cv.Size(radius, radius), 0)

  const lMod = ctx.track(new cv.Mat())
  cv.addWeighted(lFloat, 1, shadowBlurred, sAmt, 0, lMod)

  const highlightBlurred = ctx.track(new cv.Mat())
  cv.GaussianBlur(lMod, highlightBlurred, new cv.Size(radius, radius), 0)
  const lFinal = ctx.track(new cv.Mat())
  cv.addWeighted(lMod, 1, highlightBlurred, -hAmt, 0, lFinal)

  const lByte = ctx.track(new cv.Mat())
  lFinal.convertTo(lByte, cv.CV_8U)
  channels.set(0, lByte)

  const labFinal = ctx.track(new cv.Mat())
  cv.merge(channels, labFinal)
  channels.delete()

  const out = ctx.track(new cv.Mat())
  cv.cvtColor(labFinal, out, cv.COLOR_Lab2BGR)
  return { main: out }
}

export const cvAdaptiveThreshold: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv

  const gray = ctx.track(new cv.Mat())
  if (src.channels() === 1) src.copyTo(gray)
  else cv.cvtColor(src, gray, cv.COLOR_BGR2GRAY)

  let block = Math.round(Number(params.block_size) || 11)
  if (block < 3) block = 3
  if (block % 2 === 0) block += 1
  const c = Math.round(Number(params.c) ?? 5)
  const method = Number(params.adaptive_method) ?? 1
  const invert = !!params.invert

  const dst = ctx.track(new cv.Mat())
  cv.adaptiveThreshold(
    gray,
    dst,
    255,
    method === 1 ? cv.ADAPTIVE_THRESH_GAUSSIAN_C : cv.ADAPTIVE_THRESH_MEAN_C,
    invert ? cv.THRESH_BINARY : cv.THRESH_BINARY_INV,
    block,
    c
  )
  return { main: dst }
}

export const cvWhiteBalance: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, data: null }
  const cv = ctx.cv
  const bgr = ctx.track(toBgr(cv, src))

  const method = String(params.method ?? 'Gray World')
  const data = bgr.data as Uint8Array
  const n = bgr.rows * bgr.cols

  let gainB = 1, gainG = 1, gainR = 1

  if (method === 'Gray World') {
    let sumB = 0, sumG = 0, sumR = 0
    for (let i = 0; i < data.length; i += 3) {
      sumB += data[i]
      sumG += data[i + 1]
      sumR += data[i + 2]
    }
    const meanB = sumB / n + 1e-6
    const meanG = sumG / n + 1e-6
    const meanR = sumR / n + 1e-6
    const gray = (meanB + meanG + meanR) / 3
    gainB = gray / meanB
    gainG = gray / meanG
    gainR = gray / meanR
  } else if (method === 'White Patch') {
    const p = Math.max(90, Math.min(100, Number(params.percentile) || 99))
    const bArr = new Uint8Array(n)
    const gArr = new Uint8Array(n)
    const rArr = new Uint8Array(n)
    for (let i = 0, j = 0; i < data.length; i += 3, j++) {
      bArr[j] = data[i]
      gArr[j] = data[i + 1]
      rArr[j] = data[i + 2]
    }
    const percentile = (arr: Uint8Array) => {
      const sorted = Array.from(arr).sort((a, b) => a - b)
      const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length))
      return sorted[idx]
    }
    const refB = percentile(bArr) + 1e-6
    const refG = percentile(gArr) + 1e-6
    const refR = percentile(rArr) + 1e-6
    gainB = 255 / refB
    gainG = 255 / refG
    gainR = 255 / refR
  } else {
    const temp = (Number(params.temp) || 0) / 100
    const tint = (Number(params.tint) || 0) / 100
    gainR = 1 + 0.5 * temp
    gainB = 1 - 0.5 * temp
    gainG = 1 - 0.5 * tint
    gainR = gainR * (1 + 0.25 * tint)
    gainB = gainB * (1 + 0.25 * tint)
  }

  const out = ctx.track(new cv.Mat())
  bgr.copyTo(out)
  const outData = out.data as Uint8Array
  for (let i = 0; i < outData.length; i += 3) {
    outData[i] = Math.max(0, Math.min(255, Math.round(outData[i] * gainB)))
    outData[i + 1] = Math.max(0, Math.min(255, Math.round(outData[i + 1] * gainG)))
    outData[i + 2] = Math.max(0, Math.min(255, Math.round(outData[i + 2] * gainR)))
  }

  return {
    main: out,
    data: {
      method,
      channel_gains: [Number(gainB.toFixed(4)), Number(gainG.toFixed(4)), Number(gainR.toFixed(4))],
    },
  }
}
