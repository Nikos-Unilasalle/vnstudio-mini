import type { NodeImpl } from '../types'
import { toBgr, toGray } from '../cvUtils'
import { applyColormap, viridisColor } from '../colormaps'

function gaussianRandom(): number {
  let u = 0
  let v = 0
  while (u === 0) u = Math.random()
  while (v === 0) v = Math.random()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

// ---------------------------------------------------------------------------
// Noise
// ---------------------------------------------------------------------------
export const filterNoiseGaussian: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv
  const sigma = Number(params.sigma) || 25
  const out = ctx.track(toBgr(cv, src))
  const data = out.data as Uint8Array
  for (let i = 0; i < data.length; i++) {
    data[i] = Math.max(0, Math.min(255, Math.round(data[i] + gaussianRandom() * sigma)))
  }
  return { main: out }
}

export const filterNoiseSaltPepper: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv
  const prob = (Number(params.amount) || 5) / 100
  const out = ctx.track(toBgr(cv, src))
  const data = out.data as Uint8Array
  const channels = out.channels()
  const pixels = data.length / channels
  for (let p = 0; p < pixels; p++) {
    const r = Math.random()
    if (r < prob / 2) {
      for (let c = 0; c < channels; c++) data[p * channels + c] = 0
    } else if (r > 1 - prob / 2) {
      for (let c = 0; c < channels; c++) data[p * channels + c] = 255
    }
  }
  return { main: out }
}

export const filterNoiseSpeckle: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv
  const intensity = (Number(params.intensity) || 10) / 100
  const out = ctx.track(toBgr(cv, src))
  const data = out.data as Uint8Array
  for (let i = 0; i < data.length; i++) {
    const noise = gaussianRandom() * intensity
    data[i] = Math.max(0, Math.min(255, Math.round(data[i] * (1 + noise))))
  }
  return { main: out }
}

// ---------------------------------------------------------------------------
// Gabor Filter
// ---------------------------------------------------------------------------
function gaborKernel(ksize: number, sigma: number, theta: number, lambda: number, gamma: number, psi: number): Float32Array {
  const half = Math.floor(ksize / 2)
  const kernel = new Float32Array(ksize * ksize)
  const cosT = Math.cos(theta)
  const sinT = Math.sin(theta)
  for (let y = -half; y <= half; y++) {
    for (let x = -half; x <= half; x++) {
      const xTheta = x * cosT + y * sinT
      const yTheta = -x * sinT + y * cosT
      const envelope = Math.exp(-0.5 * (xTheta * xTheta + gamma * gamma * yTheta * yTheta) / (sigma * sigma))
      const carrier = Math.cos((2 * Math.PI * xTheta) / lambda + psi)
      // Matches cv2.getGaborKernel's own storage order: kernel.at(half-y, half-x).
      kernel[(half - y) * ksize + (half - x)] = envelope * carrier
    }
  }
  return kernel
}

export const filterGabor: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null, kernel: null }
  const cv = ctx.cv

  let ksize = Math.round(Number(params.ksize) || 31)
  if (ksize < 3) ksize = 3
  if (ksize % 2 === 0) ksize += 1
  const sigma = Number(params.sigma) || 4.0
  const theta = ((Number(params.theta_deg) || 0) * Math.PI) / 180
  const lambda = Number(params.lambda) || 10.0
  const gamma = Number(params.gamma) || 0.5
  const psi = ((Number(params.psi_deg) || 0) * Math.PI) / 180
  const showKernel = !!params.show_kernel

  const kernelData = gaborKernel(ksize, sigma, theta, lambda, gamma, psi)
  const kernelMat = cv.matFromArray(ksize, ksize, cv.CV_32F, kernelData)

  const gray = ctx.track(toGray(cv, src))
  const grayF = ctx.track(new cv.Mat())
  gray.convertTo(grayF, cv.CV_32F)
  const filtered = ctx.track(new cv.Mat())
  cv.filter2D(grayF, filtered, cv.CV_32F, kernelMat, new cv.Point(-1, -1), 0, cv.BORDER_DEFAULT)

  const normed = ctx.track(new cv.Mat())
  cv.normalize(filtered, normed, 0, 255, cv.NORM_MINMAX)
  const asByte = ctx.track(new cv.Mat())
  normed.convertTo(asByte, cv.CV_8U)
  const resultBgr = ctx.track(new cv.Mat())
  cv.cvtColor(asByte, resultBgr, cv.COLOR_GRAY2BGR)

  let kernelBgr: any = null
  if (showKernel) {
    const kNorm = ctx.track(new cv.Mat())
    cv.normalize(kernelMat, kNorm, 0, 255, cv.NORM_MINMAX)
    const kByte = ctx.track(new cv.Mat())
    kNorm.convertTo(kByte, cv.CV_8U)
    const scale = Math.max(1, Math.floor(256 / Math.max(1, ksize)))
    const kResized = ctx.track(new cv.Mat())
    cv.resize(kByte, kResized, new cv.Size(ksize * scale, ksize * scale), 0, 0, cv.INTER_NEAREST)
    kernelBgr = ctx.track(applyColormap(cv, kResized, viridisColor))
  }
  kernelMat.delete()

  return { main: resultBgr, kernel: kernelBgr }
}

// ---------------------------------------------------------------------------
// High / Low pass
// ---------------------------------------------------------------------------
export const filterHighPass: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv

  let kSize = Math.round(Number(params.kernel_size) || 5)
  if (kSize % 2 === 0) kSize += 1
  kSize = Math.max(1, kSize)
  const gain = Number(params.gain) || 1.0
  const useBias = params.bias !== false

  const bgr = ctx.track(toBgr(cv, src))
  const blur = ctx.track(new cv.Mat())
  cv.GaussianBlur(bgr, blur, new cv.Size(kSize, kSize), 0)

  const bgrF = ctx.track(new cv.Mat())
  const blurF = ctx.track(new cv.Mat())
  bgr.convertTo(bgrF, cv.CV_32F)
  blur.convertTo(blurF, cv.CV_32F)

  const highPass = ctx.track(new cv.Mat())
  cv.subtract(bgrF, blurF, highPass)
  const scaled = ctx.track(new cv.Mat())
  highPass.convertTo(scaled, cv.CV_32F, gain, useBias ? 128 : 0)

  const result = ctx.track(new cv.Mat())
  scaled.convertTo(result, cv.CV_8U)
  return { main: result }
}

export const filterLowPass: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv
  let kSize = Math.round(Number(params.kernel_size) || 5)
  if (kSize % 2 === 0) kSize += 1
  kSize = Math.max(1, kSize)
  const sigma = Number(params.sigma) || 1.0
  const bgr = ctx.track(toBgr(cv, src))
  const result = ctx.track(new cv.Mat())
  cv.GaussianBlur(bgr, result, new cv.Size(kSize, kSize), sigma)
  return { main: result }
}

// ---------------------------------------------------------------------------
// Laplacian
// ---------------------------------------------------------------------------
const LAPLACIAN_KERNELS: Record<number, Float32Array> = {
  0: new Float32Array([0, 1, 0, 1, -4, 1, 0, 1, 0]),
  1: new Float32Array([1, 1, 1, 1, -8, 1, 1, 1, 1]),
  2: new Float32Array([0.05, 0.2, 0.05, 0.2, -1.0, 0.2, 0.05, 0.2, 0.05]),
}

export const filterLaplacian: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { laplacian: null }
  const cv = ctx.cv

  const gray = ctx.track(toGray(cv, src))
  const grayF = ctx.track(new cv.Mat())
  gray.convertTo(grayF, cv.CV_32F, 1 / 255, 0)

  const kernelIdx = Number(params.kernel) || 0
  const kernelData = LAPLACIAN_KERNELS[kernelIdx] ?? LAPLACIAN_KERNELS[0]
  const kernelMat = cv.matFromArray(3, 3, cv.CV_32F, kernelData)
  const scale = Number(params.scale) || 1.0

  const lap = ctx.track(new cv.Mat())
  cv.filter2D(grayF, lap, cv.CV_32F, kernelMat, new cv.Point(-1, -1), 0, cv.BORDER_DEFAULT)
  kernelMat.delete()

  if (scale !== 1.0) {
    const scaled = ctx.track(new cv.Mat())
    lap.convertTo(scaled, cv.CV_32F, scale, 0)
    if (params.normalize) {
      const normed = ctx.track(new cv.Mat())
      cv.normalize(scaled, normed, 0, 1, cv.NORM_MINMAX)
      return { laplacian: normed }
    }
    return { laplacian: scaled }
  }

  if (params.normalize) {
    const normed = ctx.track(new cv.Mat())
    cv.normalize(lap, normed, 0, 1, cv.NORM_MINMAX)
    return { laplacian: normed }
  }
  return { laplacian: lap }
}

// ---------------------------------------------------------------------------
// Image Clamp
// ---------------------------------------------------------------------------
export const filterImgClamp: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { image: null }
  const cv = ctx.cv
  const mn = Number(params.min_val) ?? 0
  const mx = Number(params.max_val) ?? 1
  const out = ctx.track(src.clone())
  const isFloat = src.depth() === cv.CV_32F || src.depth() === cv.CV_64F
  if (isFloat) {
    const data = (src.depth() === cv.CV_32F ? out.data32F : out.data64F) as Float32Array | Float64Array
    for (let i = 0; i < data.length; i++) data[i] = Math.max(mn, Math.min(mx, data[i]))
  } else {
    const bytes = out.data as Uint8Array
    const lo = Math.max(0, Math.round(mn))
    const hi = Math.min(255, Math.round(mx))
    for (let i = 0; i < bytes.length; i++) bytes[i] = Math.max(lo, Math.min(hi, bytes[i]))
  }
  return { image: out }
}

// ---------------------------------------------------------------------------
// Gradient (Sobel / Scharr)
// ---------------------------------------------------------------------------
export const pluginGradient: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { magnitude: null, angle: null, dx: null, dy: null }
  const cv = ctx.cv
  const gray = ctx.track(toGray(cv, src))

  const method = Number(params.method) || 0
  let ksize = Math.round(Number(params.ksize) || 3)
  const doNorm = params.normalize !== false

  const dx = ctx.track(new cv.Mat())
  const dy = ctx.track(new cv.Mat())
  if (method === 0) {
    if (ksize % 2 === 0) ksize += 1
    cv.Sobel(gray, dx, cv.CV_64F, 1, 0, ksize)
    cv.Sobel(gray, dy, cv.CV_64F, 0, 1, ksize)
  } else {
    cv.Scharr(gray, dx, cv.CV_64F, 1, 0)
    cv.Scharr(gray, dy, cv.CV_64F, 0, 1)
  }

  const magnitude = ctx.track(new cv.Mat())
  const angle = ctx.track(new cv.Mat())
  cv.cartToPolar(dx, dy, magnitude, angle, true)

  if (doNorm) {
    const magVis = ctx.track(new cv.Mat())
    cv.convertScaleAbs(magnitude, magVis)
    const angData = angle.data64F as Float64Array
    const angVis = ctx.track(new cv.Mat(angle.rows, angle.cols, cv.CV_8U))
    const angVisData = angVis.data as Uint8Array
    for (let i = 0; i < angData.length; i++) angVisData[i] = Math.round(angData[i] * (255 / 360))
    return { magnitude: magVis, angle: angVis, dx, dy }
  }

  return { magnitude, angle, dx, dy }
}

// ---------------------------------------------------------------------------
// Channel Split / Merge
// ---------------------------------------------------------------------------
export const pluginChannelSplit: NodeImpl = (inputs, _params, ctx) => {
  const src = inputs.image as any
  if (!src) return { r: null, g: null, b: null, a: null }
  const cv = ctx.cv

  if (src.channels() === 1) {
    const asBgr = ctx.track(new cv.Mat())
    cv.cvtColor(src, asBgr, cv.COLOR_GRAY2BGR)
    return { r: asBgr, g: asBgr, b: asBgr, a: null }
  }

  const channels = new cv.MatVector()
  cv.split(src, channels)
  const toBgrOut = (ch: any) => {
    const out = ctx.track(new cv.Mat())
    cv.cvtColor(ch, out, cv.COLOR_GRAY2BGR)
    return out
  }
  const b = toBgrOut(channels.get(0))
  const g = toBgrOut(channels.get(1))
  const r = toBgrOut(channels.get(2))
  const a = channels.size() === 4 ? toBgrOut(channels.get(3)) : null
  channels.delete()
  return { r, g, b, a }
}

export const pluginChannelMerge: NodeImpl = (inputs, _params, ctx) => {
  const cv = ctx.cv
  const toGrayIn = (img: any) => {
    if (!img) return null
    return img.channels() === 1 ? img : ctx.track(toGray(cv, img))
  }
  const r = toGrayIn(inputs.r)
  const g = toGrayIn(inputs.g)
  const b = toGrayIn(inputs.b)
  const a = toGrayIn(inputs.a)

  const ref = r ?? g ?? b
  if (!ref) return { main: null }
  const h = ref.rows
  const w = ref.cols

  const fit = (ch: any) => {
    if (!ch) return ctx.track(new cv.Mat(h, w, cv.CV_8U, new cv.Scalar(0)))
    if (ch.rows === h && ch.cols === w) return ch
    const resized = ctx.track(new cv.Mat())
    cv.resize(ch, resized, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR)
    return resized
  }

  const channels = new cv.MatVector()
  channels.push_back(fit(b))
  channels.push_back(fit(g))
  channels.push_back(fit(r))
  if (a) channels.push_back(fit(a))

  const merged = ctx.track(new cv.Mat())
  cv.merge(channels, merged)
  channels.delete()
  return { main: merged }
}

// ---------------------------------------------------------------------------
// Blend
// ---------------------------------------------------------------------------
function blendPixel(mode: number, a: number, b: number): number {
  switch (mode) {
    case 0:
      return b
    case 1:
      return a * b
    case 2:
      return 1 - (1 - a) * (1 - b)
    case 3:
      return a < 0.5 ? 2 * a * b : 1 - 2 * (1 - a) * (1 - b)
    case 4:
      return b < 0.5 ? 2 * a * b : 1 - 2 * (1 - a) * (1 - b)
    case 5:
      return (1 - 2 * b) * a * a + 2 * b * a
    case 6:
      return 1 - a === 0 ? 1 : Math.min(1, b / (1 - a))
    case 7:
      return a === 0 ? 1 : 1 - Math.min(1, (1 - b) / a)
    case 8:
      return a + b
    case 9:
      return a + b - 1
    case 10:
      return b < 0.5 ? (2 * b === 0 ? 1 : 1 - (1 - a) / (2 * b)) : (2 * (1 - b) === 0 ? 1 : a / (2 * (1 - b)))
    case 11:
      return a + 2 * b - 1
    case 12:
      return b < 0.5 ? Math.min(a, 2 * b) : Math.max(a, 2 * b - 1)
    case 13:
      return a + b >= 1 ? 1 : 0
    case 14:
      return Math.abs(a - b)
    case 15:
      return a + b - 2 * a * b
    case 16:
      return b - a
    case 17:
      return a === 0 ? 1 : Math.min(1, b / a)
    case 18:
      return a - b + 0.5
    case 19:
      return a + b - 0.5
    case 20:
      return Math.min(a, b)
    case 21:
      return Math.max(a, b)
    default:
      return b
  }
}

export const pluginBlendModes: NodeImpl = (inputs, params, ctx) => {
  const imgA = inputs.image_a as any
  const imgB = inputs.image_b as any
  if (!imgA) return { main: imgB ?? null }
  if (!imgB) return { main: imgA }
  const cv = ctx.cv

  const a = ctx.track(toBgr(cv, imgA))
  let b = ctx.track(toBgr(cv, imgB))
  if (b.cols !== a.cols || b.rows !== a.rows) {
    const resized = ctx.track(new cv.Mat())
    cv.resize(b, resized, new cv.Size(a.cols, a.rows), 0, 0, cv.INTER_LINEAR)
    b = resized
  }

  const mode = Number(params.mode) || 0
  const opacity = (Number(params.opacity) ?? 50) / 100

  const out = ctx.track(new cv.Mat(a.rows, a.cols, cv.CV_8UC3))
  const aData = a.data as Uint8Array
  const bData = b.data as Uint8Array
  const outData = out.data as Uint8Array
  for (let i = 0; i < aData.length; i++) {
    const av = aData[i] / 255
    const bv = bData[i] / 255
    const res = blendPixel(mode, av, bv)
    const blended = av * (1 - opacity) + res * opacity
    outData[i] = Math.max(0, Math.min(255, Math.round(blended * 255)))
  }

  return { main: out }
}

// ---------------------------------------------------------------------------
// Pixelate / Glitch
// ---------------------------------------------------------------------------
export const pluginPixelate: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv
  const blocks = Math.max(1, Math.round(Number(params.blocks) || 10))
  const w = src.cols
  const h = src.rows
  const smallW = Math.max(1, Math.floor(w / blocks))
  const smallH = Math.max(1, Math.floor(h / blocks))

  const small = ctx.track(new cv.Mat())
  cv.resize(src, small, new cv.Size(smallW, smallH), 0, 0, cv.INTER_LINEAR)
  const out = ctx.track(new cv.Mat())
  cv.resize(small, out, new cv.Size(w, h), 0, 0, cv.INTER_NEAREST)
  return { main: out }
}

export const filterGlitch: NodeImpl = (inputs, params, ctx) => {
  const src = inputs.image as any
  if (!src) return { main: null }
  const cv = ctx.cv
  const intensity = Math.round(Number(params.intensity) || 20)
  const shift = Math.round(Number(params.shift) || 10)
  if (intensity === 0) return { main: src }

  const out = ctx.track(toBgr(cv, src))
  const w = out.cols
  const h = out.rows
  const channels = out.channels()
  const data = out.data as Uint8Array

  const rollRow = (y: number, xStart: number, xEnd: number, s: number, ch: number) => {
    const rowLen = xEnd - xStart
    if (rowLen <= 0) return
    const copy = new Uint8Array(rowLen)
    for (let x = 0; x < rowLen; x++) copy[x] = data[(y * w + xStart + x) * channels + ch]
    for (let x = 0; x < rowLen; x++) {
      const srcX = ((((x - s) % rowLen) + rowLen) % rowLen) + xStart
      data[(y * w + xStart + x) * channels + ch] = copy[srcX - xStart]
    }
  }

  if (Math.random() < intensity / 100) {
    const channel = Math.floor(Math.random() * 3)
    const s = Math.round((Math.random() * 2 - 1) * shift)
    for (let y = 0; y < h; y++) rollRow(y, 0, w, s, channel)
  }

  for (let i = 0; i < Math.floor(intensity / 10); i++) {
    if (Math.random() < 0.5) {
      const y = Math.floor(Math.random() * h)
      const hSlice = Math.max(1, Math.floor(Math.random() * 10) + 1)
      const s = Math.round((Math.random() * 2 - 1) * shift * 2)
      for (let dy = 0; dy < hSlice && y + dy < h; dy++) {
        for (let ch = 0; ch < channels; ch++) rollRow(y + dy, 0, w, s, ch)
      }
    }
  }

  return { main: out }
}
