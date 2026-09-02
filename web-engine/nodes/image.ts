import type { NodeImpl } from '../types'
import { toBgr, toGray } from '../cvUtils'

export const filterGray: NodeImpl = (inputs, _params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  return { main: ctx.track(toGray(ctx.cv, src)) }
}

export const filterBlur: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv
  // Kernel sizes must be odd for Gaussian/Median; round the slider up rather than throwing.
  const size = Math.max(1, Number(params.size) || 5) | 1
  const dst = ctx.track(new cv.Mat())
  const method = Number(params.method) || 0

  if (method === 1) {
    cv.medianBlur(src, dst, size)
  } else if (method === 2) {
    cv.blur(src, dst, new cv.Size(size, size))
  } else if (method === 3) {
    const bgr = ctx.track(toBgr(cv, src))
    cv.bilateralFilter(bgr, dst, size, Number(params.sigma_color) || 75, Number(params.sigma_space) || 75)
  } else {
    cv.GaussianBlur(src, dst, new cv.Size(size, size), Number(params.sigma) || 0)
  }
  return { main: dst }
}

export const filterCanny: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv
  const gray = ctx.track(toGray(cv, src))
  const dst = ctx.track(new cv.Mat())
  cv.Canny(gray, dst, Number(params.low) || 100, Number(params.high) || 200)
  return { main: dst }
}

export const filterThreshold: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, mask: null }
  const cv = ctx.cv
  const gray = ctx.track(toGray(cv, src))
  const dst = ctx.track(new cv.Mat())
  cv.threshold(gray, dst, Number(params.threshold) || 127, 255, cv.THRESH_BINARY)
  return { main: dst, mask: dst }
}

export const pluginInvert: NodeImpl = (inputs, _params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const dst = ctx.track(new ctx.cv.Mat())
  ctx.cv.bitwise_not(src, dst)
  return { main: dst }
}

export const pluginBrightnessContrast: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const brightness = Number(params.brightness) || 0
  const contrast = Number(params.contrast) || 0
  // Matches the desktop plugin: contrast is a -100..100 slider mapped to a gain around 1.
  const alpha = contrast >= 0 ? 1 + contrast / 100 : 1 + contrast / 200
  const dst = ctx.track(new ctx.cv.Mat())
  src.convertTo(dst, -1, alpha, brightness)
  return { main: dst }
}

export const pluginSobel: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv
  const gray = ctx.track(toGray(cv, src))
  const ksize = Math.max(1, Number(params.kernel_size) || 3) | 1
  const useX = params.x_dir !== false
  const useY = params.y_dir !== false

  const gx = ctx.track(new cv.Mat())
  const gy = ctx.track(new cv.Mat())
  if (useX) cv.Sobel(gray, gx, cv.CV_32F, 1, 0, ksize)
  else cv.Mat.zeros(gray.rows, gray.cols, cv.CV_32F).copyTo(gx)
  if (useY) cv.Sobel(gray, gy, cv.CV_32F, 0, 1, ksize)
  else cv.Mat.zeros(gray.rows, gray.cols, cv.CV_32F).copyTo(gy)

  const mag = ctx.track(new cv.Mat())
  cv.magnitude(gx, gy, mag)
  const norm = ctx.track(new cv.Mat())
  cv.normalize(mag, norm, 0, 255, cv.NORM_MINMAX)
  const out = ctx.track(new cv.Mat())
  norm.convertTo(out, cv.CV_8U)
  return { main: out }
}

export const cvColorspace: NodeImpl = (inputs, _params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, hsv: null, lab: null }
  const cv = ctx.cv
  const bgr = ctx.track(toBgr(cv, src))
  const hsv = ctx.track(new cv.Mat())
  const lab = ctx.track(new cv.Mat())
  cv.cvtColor(bgr, hsv, cv.COLOR_BGR2HSV)
  cv.cvtColor(bgr, lab, cv.COLOR_BGR2Lab)
  return { main: bgr, hsv, lab }
}

export const featClahe: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, luma: null }
  const cv = cv_(ctx)
  const clipLimit = Number(params.clip_limit) || 2
  const gridSize = Math.max(1, Number(params.grid_size) || 8)
  const clahe = new cv.CLAHE(clipLimit, new cv.Size(gridSize, gridSize))

  const colorSpace = Number(params.color_space) || 0
  if (src.channels() === 1 || colorSpace === 4) {
    const gray = ctx.track(toGray(cv, src))
    const out = ctx.track(new cv.Mat())
    clahe.apply(gray, out)
    clahe.delete()
    return { main: out, luma: out }
  }

  const bgr = ctx.track(toBgr(cv, src))
  const converted = ctx.track(new cv.Mat())
  // Equalising luminance alone avoids the colour casts that per-channel CLAHE causes.
  const [toSpace, fromSpace] =
    colorSpace === 1
      ? [cv.COLOR_BGR2YCrCb, cv.COLOR_YCrCb2BGR]
      : colorSpace === 2
        ? [cv.COLOR_BGR2HSV, cv.COLOR_HSV2BGR]
        : [cv.COLOR_BGR2Lab, cv.COLOR_Lab2BGR]
  cv.cvtColor(bgr, converted, toSpace)

  const channels = new cv.MatVector()
  cv.split(converted, channels)
  const lumaIndex = colorSpace === 2 ? 2 : 0
  const luma = channels.get(lumaIndex)
  const equalized = new cv.Mat()
  clahe.apply(luma, equalized)
  channels.set(lumaIndex, equalized)

  const merged = ctx.track(new cv.Mat())
  cv.merge(channels, merged)
  const out = ctx.track(new cv.Mat())
  cv.cvtColor(merged, out, fromSpace)

  const lumaOut = ctx.track(equalized.clone())
  equalized.delete()
  channels.delete()
  clahe.delete()
  return { main: out, luma: lumaOut }
}

function cv_(ctx: { cv: any }): any {
  return ctx.cv
}
