import type { NodeImpl } from '../types'
import { toBgr, toGray } from '../cvUtils'
import { applyColormap, infernoColor, viridisColor } from '../colormaps'
import { dct2d, fft2d, fftShift, idct2d, ifftShift } from '../fft'

/**
 * The spectral payload passed between FFT, Spectral Gain and Inverse FFT.
 * Marked so the executor can recognise it, and carrying plain typed arrays
 * rather than Mats so it survives being handed from node to node.
 */
export interface SpectrumPayload {
  _spectrum: true
  channels: { re: Float64Array; im: Float64Array }[]
  isColor: boolean
  width: number
  height: number
}

export interface MagnitudePayload {
  _spectrum: true
  channels: Float64Array[]
  isColor: boolean
  width: number
  height: number
}

const FILTER_TYPES = ['None', 'Low-pass', 'High-pass', 'Band-pass', 'Band-stop']

/** Stretches a float field into a byte image, as cv2.normalize NORM_MINMAX does. */
function toByteMat(cv: any, data: Float64Array, w: number, h: number, preserveRange: boolean): any {
  const out = new cv.Mat(h, w, cv.CV_8U)
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < data.length; i++) {
    if (data[i] < lo) lo = data[i]
    if (data[i] > hi) hi = data[i]
  }
  if (preserveRange) {
    // Scientific range: keep zero at zero so relative intensities stay readable.
    const scale = hi > 0 ? 255 / hi : 0
    for (let i = 0; i < data.length; i++) out.data[i] = Math.max(0, Math.min(255, Math.round(data[i] * scale)))
  } else {
    const span = hi - lo || 1
    for (let i = 0; i < data.length; i++) out.data[i] = Math.round(((data[i] - lo) / span) * 255)
  }
  return out
}

export const sciFft: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const source = (inputs.image ?? inputs.main) as any
  if (!source) return { main: null, magnitude: null, phase: null, complex_data: null, magnitude_raw: null }

  const isColor = source.channels() >= 3
  const image = isColor ? toBgr(cv, source) : toGray(cv, source)
  const w = image.cols
  const h = image.rows
  const n = w * h
  const planes = isColor ? 3 : 1

  const filterType = FILTER_TYPES[Math.round(Number(params.filter_type) || 0)] ?? 'None'
  const lowNorm = (Number(params.low_cutoff) || 0) / 100
  const highNorm = (Number(params.high_cutoff) ?? 10) / 100
  const logScale = params.log_scale !== false
  const preserve = !!params.preserve_dynamic_range

  // The filter is a radial mask on the centred spectrum, so a cutoff is a
  // fraction of the half-diagonal — the largest frequency the image can hold.
  const mask = new Float64Array(n).fill(1)
  if (filterType !== 'None') {
    const cx = Math.floor(w / 2)
    const cy = Math.floor(h / 2)
    const diagonal = Math.sqrt(w * w + h * h) / 2
    const rInner = Math.round(lowNorm * diagonal)
    const rOuter = Math.round(highNorm * diagonal)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const r = Math.hypot(x - cx, y - cy)
        let value: number
        if (filterType === 'Low-pass') value = r <= rOuter ? 1 : 0
        else if (filterType === 'High-pass') value = r <= rInner ? 0 : 1
        else if (filterType === 'Band-pass') value = r <= rOuter && r > rInner ? 1 : 0
        else value = r <= rOuter && r > rInner ? 0 : 1
        mask[y * w + x] = value
      }
    }
  }

  const magnitudeVis: Float64Array[] = []
  const phaseVis: Float64Array[] = []
  const filtered: Float64Array[] = []
  const complexChannels: { re: Float64Array; im: Float64Array }[] = []
  const magnitudeRaw: Float64Array[] = []

  const data = image.data
  for (let c = 0; c < planes; c++) {
    const re = new Float64Array(n)
    const im = new Float64Array(n)
    for (let i = 0; i < n; i++) re[i] = data[i * planes + c]
    fft2d(re, im, w, h, false)

    const shiftedRe = fftShift(re, w, h)
    const shiftedIm = fftShift(im, w, h)

    const magnitude = new Float64Array(n)
    const phase = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      magnitude[i] = Math.hypot(shiftedRe[i], shiftedIm[i])
      phase[i] = Math.atan2(shiftedIm[i], shiftedRe[i])
    }
    magnitudeRaw.push(magnitude)

    const filteredRe = new Float64Array(n)
    const filteredIm = new Float64Array(n)
    for (let i = 0; i < n; i++) {
      filteredRe[i] = shiftedRe[i] * mask[i]
      filteredIm[i] = shiftedIm[i] * mask[i]
    }
    complexChannels.push({ re: filteredRe, im: filteredIm })

    const backRe = ifftShift(filteredRe, w, h)
    const backIm = ifftShift(filteredIm, w, h)
    fft2d(backRe, backIm, w, h, true)
    const reconstructed = new Float64Array(n)
    for (let i = 0; i < n; i++) reconstructed[i] = Math.hypot(backRe[i], backIm[i])
    filtered.push(reconstructed)

    const vis = new Float64Array(n)
    // The spectrum spans many orders of magnitude, so without the log only the
    // DC term is visible.
    for (let i = 0; i < n; i++) vis[i] = logScale ? Math.log(magnitude[i] + 1) : magnitude[i]
    magnitudeVis.push(vis)

    const phaseBytes = new Float64Array(n)
    for (let i = 0; i < n; i++) phaseBytes[i] = ((phase[i] + Math.PI) / (2 * Math.PI)) * 255
    phaseVis.push(phaseBytes)
  }

  /** Interleaves the per-channel planes back into one Mat. */
  const stack = (list: Float64Array[], normalise: boolean) => {
    if (list.length === 1) return toByteMat(cv, list[0], w, h, !normalise)
    const bytes = list.map((plane) => toByteMat(cv, plane, w, h, !normalise))
    const merged = new cv.Mat(h, w, cv.CV_8UC3)
    for (let i = 0; i < n; i++) {
      merged.data[i * 3] = bytes[0].data[i]
      merged.data[i * 3 + 1] = bytes[1].data[i]
      merged.data[i * 3 + 2] = bytes[2].data[i]
    }
    for (const b of bytes) b.delete()
    return merged
  }

  const outFiltered = ctx.track(stack(filtered, !preserve))
  const outPhase = ctx.track(stack(phaseVis, false))

  const magStacked = stack(magnitudeVis, !preserve)
  const magGray = magStacked.channels() === 1 ? magStacked : toGray(cv, magStacked)
  const outMagnitude = ctx.track(applyColormap(cv, magGray, infernoColor))
  if (magGray !== magStacked) magGray.delete()
  magStacked.delete()
  image.delete()

  return {
    main: outFiltered,
    magnitude: outMagnitude,
    phase: outPhase,
    complex_data: { _spectrum: true, channels: complexChannels, isColor, width: w, height: h } satisfies SpectrumPayload,
    magnitude_raw: { _spectrum: true, channels: magnitudeRaw, isColor, width: w, height: h } satisfies MagnitudePayload,
  }
}

export const sciIfft: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const complexData = inputs.complex_data as SpectrumPayload | null | undefined
  const magnitudeRaw = inputs.magnitude_raw as MagnitudePayload | null | undefined
  const magnitudeImage = inputs.magnitude as any
  const phaseImage = inputs.phase as any

  const modes = ['auto', 'complex_data', 'magnitude_phase']
  const mode = modes[Math.round(Number(params.mode) || 0)] ?? 'auto'
  const preserve = params.preserve_dynamic_range !== false
  const inverseLog = !!params.inv_log

  let planes: Float64Array[] = []
  let isColor = false
  let w = 0
  let h = 0

  if ((mode === 'complex_data' || mode === 'auto') && complexData?.channels?.length) {
    isColor = complexData.isColor
    w = complexData.width
    h = complexData.height
    for (const channel of complexData.channels) {
      const re = ifftShift(channel.re, w, h)
      const im = ifftShift(channel.im, w, h)
      fft2d(re, im, w, h, true)
      const out = new Float64Array(w * h)
      for (let i = 0; i < out.length; i++) out[i] = Math.hypot(re[i], im[i])
      planes.push(out)
    }
  } else {
    // Magnitude and phase: perfect only from the raw magnitude, since the
    // magnitude *image* has been log-scaled and quantised for display.
    let magnitudes: Float64Array[] = []
    let fromRaw = false
    if (magnitudeRaw?.channels?.length) {
      isColor = magnitudeRaw.isColor
      w = magnitudeRaw.width
      h = magnitudeRaw.height
      magnitudes = magnitudeRaw.channels
      fromRaw = true
    } else if (magnitudeImage) {
      w = magnitudeImage.cols
      h = magnitudeImage.rows
      isColor = magnitudeImage.channels() >= 3
      const count = isColor ? 3 : 1
      for (let c = 0; c < count; c++) {
        const plane = new Float64Array(w * h)
        for (let i = 0; i < plane.length; i++) plane[i] = magnitudeImage.data[i * count + c]
        magnitudes.push(plane)
      }
    } else return { main: null }

    let phases: Float64Array[]
    if (phaseImage) {
      const count = phaseImage.channels()
      phases = magnitudes.map((_, c) => {
        const plane = new Float64Array(w * h)
        const at = Math.min(c, count - 1)
        for (let i = 0; i < plane.length; i++) {
          // Phase travels as a byte image over [-π, π].
          plane[i] = (phaseImage.data[i * count + at] / 255) * 2 * Math.PI - Math.PI
        }
        return plane
      })
    } else {
      phases = magnitudes.map(() => new Float64Array(w * h))
    }

    for (let c = 0; c < magnitudes.length; c++) {
      const magnitude = magnitudes[c]
      const phase = phases[c]
      const re = new Float64Array(w * h)
      const im = new Float64Array(w * h)
      for (let i = 0; i < re.length; i++) {
        const m = !fromRaw && inverseLog ? Math.exp(magnitude[i]) - 1 : magnitude[i]
        re[i] = m * Math.cos(phase[i])
        im[i] = m * Math.sin(phase[i])
      }
      const shiftedRe = ifftShift(re, w, h)
      const shiftedIm = ifftShift(im, w, h)
      fft2d(shiftedRe, shiftedIm, w, h, true)
      const out = new Float64Array(w * h)
      for (let i = 0; i < out.length; i++) out[i] = Math.hypot(shiftedRe[i], shiftedIm[i])
      planes.push(out)
    }
  }

  if (!planes.length) return { main: null }

  const bytes = planes.map((plane) => toByteMat(cv, plane, w, h, preserve))
  let out: any
  if (isColor && bytes.length === 3) {
    out = ctx.track(new cv.Mat(h, w, cv.CV_8UC3))
    for (let i = 0; i < w * h; i++) {
      out.data[i * 3] = bytes[0].data[i]
      out.data[i * 3 + 1] = bytes[1].data[i]
      out.data[i * 3 + 2] = bytes[2].data[i]
    }
    for (const b of bytes) b.delete()
  } else {
    out = ctx.track(bytes[0])
    for (let i = 1; i < bytes.length; i++) bytes[i].delete()
  }
  return { main: out }
}

export const sciSpectralGain: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const payload = inputs.magnitude_raw as MagnitudePayload | null | undefined
  if (!payload?.channels?.length) return { magnitude_raw: null, preview: null }

  const w = payload.width
  const h = payload.height
  const lowGain = (Number(params.low_gain) ?? 100) / 100
  const midGain = (Number(params.mid_gain) ?? 100) / 100
  const highGain = (Number(params.high_gain) ?? 100) / 100
  let lowMid = (Number(params.low_mid_split) ?? 15) / 100
  let midHigh = (Number(params.mid_high_split) ?? 50) / 100
  // The splits must stay ordered or the middle band would vanish.
  if (lowMid >= midHigh) midHigh = Math.min(lowMid + 0.01, 1)
  if (lowMid <= 0) lowMid = 0.01
  if (midHigh >= 1) midHigh = 0.99

  const cx = Math.floor(w / 2)
  const cy = Math.floor(h / 2)
  const diagonal = Math.sqrt(w * w + h * h) / 2
  const gain = new Float64Array(w * h)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      // Radial distance from DC, normalised: 0 at the centre, 1 at a corner.
      const d = Math.hypot(x - cx, y - cy) / diagonal
      gain[y * w + x] = d <= lowMid ? lowGain : d <= midHigh ? midGain : highGain
    }
  }

  const boosted = payload.channels.map((channel) => {
    const out = new Float64Array(channel.length)
    for (let i = 0; i < out.length; i++) out[i] = channel[i] * gain[i]
    return out
  })

  const vis = new Float64Array(w * h)
  for (let i = 0; i < vis.length; i++) vis[i] = Math.log(Math.abs(boosted[0][i]) + 1)
  const eight = toByteMat(cv, vis, w, h, false)
  const preview = ctx.track(applyColormap(cv, eight, infernoColor))
  eight.delete()

  return {
    magnitude_raw: { _spectrum: true, channels: boosted, isColor: payload.isColor, width: w, height: h } satisfies MagnitudePayload,
    preview,
  }
}

export const sciDct: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const source = (inputs.image ?? inputs.main) as any
  if (!source) return { main: null, data: null }

  const gray = toGray(cv, source)
  const h = gray.rows
  const w = gray.cols
  // The desktop pads to even dimensions because cv2.dct requires them; the
  // padding is kept so the coefficient grid matches, then cropped away.
  const ph = h + (h % 2)
  const pw = w + (w % 2)
  const padded = new Float64Array(ph * pw)
  for (let y = 0; y < ph; y++) {
    const sy = Math.min(y, h - 1)
    for (let x = 0; x < pw; x++) {
      padded[y * pw + x] = gray.data[sy * w + Math.min(x, w - 1)]
    }
  }

  const coefficients = dct2d(padded, pw, ph)
  let totalEnergy = 0
  for (let i = 0; i < coefficients.length; i++) totalEnergy += coefficients[i] * coefficients[i]

  const keep = Math.max(1, Math.round(Number(params.keep_coeffs) || 32))
  const kh = Math.min(keep, ph)
  const kw = Math.min(keep, pw)
  let keptEnergy = 0
  for (let y = 0; y < kh; y++) for (let x = 0; x < kw; x++) keptEnergy += coefficients[y * pw + x] ** 2

  const normalise = params.normalize !== false
  const spectrumMode = String(params.output ?? 'Log Spectrum') !== 'Reconstruction' && Math.round(Number(params.output) || 0) === 0

  let result: any
  if (spectrumMode) {
    const vis = new Float64Array(coefficients.length)
    for (let i = 0; i < vis.length; i++) vis[i] = Math.log1p(Math.abs(coefficients[i]))
    const eight = toByteMat(cv, vis, pw, ph, !normalise)
    result = applyColormap(cv, eight, viridisColor)
    eight.delete()
  } else {
    // Keeping only the top-left block is exactly what JPEG's quantisation
    // approximates, so the reconstruction shows the compression artefacts.
    const kept = new Float64Array(coefficients.length)
    for (let y = 0; y < kh; y++) for (let x = 0; x < kw; x++) kept[y * pw + x] = coefficients[y * pw + x]
    const recon = idct2d(kept, pw, ph)
    const eight = toByteMat(cv, recon, pw, ph, !normalise)
    result = new cv.Mat()
    cv.cvtColor(eight, result, cv.COLOR_GRAY2BGR)
    eight.delete()
  }

  const cropped = ctx.track(result.roi(new cv.Rect(0, 0, w, h)).clone())
  result.delete()
  gray.delete()

  return {
    main: cropped,
    data: {
      energy_kept_ratio: totalEnergy > 0 ? keptEnergy / totalEnergy : 0,
      dims: [h, w],
    },
  }
}
