import type { NodeImpl } from '../types'
import { toGray } from '../cvUtils'
import { applyColormap, COLORMAPS } from '../colormaps'

/** Small deterministic PRNG (mulberry32) — a seeded stand-in for numpy's default_rng. */
function makeRng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const PATTERNS = [
  'solid', 'center_dot', 'white_noise', 'value_noise',
  'checkerboard', 'stripes_h', 'stripes_v', 'dots_grid',
  'gradient_h', 'gradient_v', 'gradient_radial', 'rings',
]

function valueNoise(cv: any, w: number, h: number, scale: number, octaves: number, seed: number): Float32Array {
  const result = new Float32Array(w * h)
  let amp = 1
  let freq = scale
  let totalAmp = 0
  for (let o = 0; o < octaves; o++) {
    const gw = Math.max(2, Math.round(w * freq))
    const gh = Math.max(2, Math.round(h * freq))
    const rng = makeRng(seed + o * 7919)
    const small = new cv.Mat(gh, gw, cv.CV_32F)
    const smallData = small.data32F as Float32Array
    for (let i = 0; i < smallData.length; i++) smallData[i] = rng()
    const resized = new cv.Mat()
    cv.resize(small, resized, new cv.Size(w, h), 0, 0, cv.INTER_CUBIC)
    const resizedData = resized.data32F as Float32Array
    for (let i = 0; i < result.length; i++) result[i] += resizedData[i] * amp
    totalAmp += amp
    small.delete()
    resized.delete()
    amp *= 0.5
    freq *= 2
  }
  for (let i = 0; i < result.length; i++) result[i] = Math.max(0, Math.min(1, result[i] / totalAmp))
  return result
}

function generatePattern(cv: any, name: string, w: number, h: number, size: number, tiles: number, octaves: number, scale: number, seed: number): Float32Array {
  const out = new Float32Array(w * h)
  const cx = w / 2
  const cy = h / 2

  if (name === 'solid') {
    out.fill(1)
  } else if (name === 'center_dot') {
    const r = Math.max(1, Math.round(Math.min(w, h) * size))
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = (x - cx) ** 2 + (y - cy) ** 2 <= r * r ? 1 : 0
  } else if (name === 'white_noise') {
    const rng = makeRng(seed)
    for (let i = 0; i < out.length; i++) out[i] = rng()
  } else if (name === 'value_noise') {
    return valueNoise(cv, w, h, scale, octaves, seed)
  } else if (name === 'checkerboard') {
    const cell = Math.max(1, Math.floor(Math.min(w, h) / tiles))
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = (Math.floor(x / cell) + Math.floor(y / cell)) % 2
  } else if (name === 'stripes_h') {
    const cell = Math.max(1, Math.floor(h / tiles))
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = Math.floor(y / cell) % 2
  } else if (name === 'stripes_v') {
    const cell = Math.max(1, Math.floor(w / tiles))
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) out[y * w + x] = Math.floor(x / cell) % 2
  } else if (name === 'dots_grid') {
    const cell = Math.max(1, Math.floor(Math.min(w, h) / tiles))
    const r = Math.max(1, Math.round(cell * size * 4))
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const cxg = (x % cell) - Math.floor(cell / 2)
        const cyg = (y % cell) - Math.floor(cell / 2)
        out[y * w + x] = cxg * cxg + cyg * cyg <= r * r ? 1 : 0
      }
    }
  } else if (name === 'gradient_h') {
    for (let x = 0; x < w; x++) {
      const v = w > 1 ? x / (w - 1) : 0
      for (let y = 0; y < h; y++) out[y * w + x] = v
    }
  } else if (name === 'gradient_v') {
    for (let y = 0; y < h; y++) {
      const v = h > 1 ? y / (h - 1) : 0
      for (let x = 0; x < w; x++) out[y * w + x] = v
    }
  } else if (name === 'gradient_radial') {
    let maxDist = 0
    const dist = new Float32Array(w * h)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const d = Math.hypot(x - cx, y - cy)
        dist[y * w + x] = d
        if (d > maxDist) maxDist = d
      }
    }
    if (maxDist === 0) maxDist = 1
    for (let i = 0; i < out.length; i++) out[i] = 1 - dist[i] / maxDist
  } else if (name === 'rings') {
    const freq = tiles / (Math.min(w, h) / 2)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const d = Math.hypot(x - cx, y - cy)
        out[y * w + x] = (Math.sin(d * freq * Math.PI) + 1) * 0.5
      }
    }
  }
  return out
}

interface CanvasState {
  canvas: any | null
  prevReset: number
  prevRegen: number
  frame: number
}

export const genCanvas: NodeImpl = (inputs, params, ctx) => {
  let state: CanvasState = ctx.state.get(ctx.nodeId)
  if (!state) {
    state = { canvas: null, prevReset: 0, prevRegen: 0, frame: 0 }
    ctx.state.set(ctx.nodeId, state)
  }
  const cv = ctx.cv

  const resetRaw = inputs.reset
  const reset = typeof resetRaw === 'number' ? resetRaw : 0
  const regen = Number(params.regenerate) || 0
  const animate = !!params.animate

  const needsBuild = !state.canvas || (reset > 0.5 && state.prevReset <= 0.5) || (regen === 1 && state.prevRegen === 0) || animate

  if (needsBuild) {
    const w = Math.round(Number(params.width) || 512)
    const h = Math.round(Number(params.height) || 512)
    const pIdx = Number(params.pattern) || 0
    const vmin = Number(params.value_min) ?? 0
    const vmax = Number(params.value_max) ?? 1
    const size = Number(params.seed_size) || 0.05
    const tiles = Math.round(Number(params.tile_count) || 8)
    const octaves = Math.round(Number(params.octaves) || 4)
    const scale = Number(params.noise_scale) || 0.1
    const seed = Math.round(Number(params.seed) || 42) + state.frame
    const name = PATTERNS[pIdx] ?? 'solid'

    const raw = generatePattern(cv, name, w, h, size, tiles, octaves, scale, seed)
    if (state.canvas) state.canvas.delete()
    const mat = new cv.Mat(h, w, cv.CV_32F)
    const data = mat.data32F as Float32Array
    for (let i = 0; i < data.length; i++) data[i] = Math.max(0, Math.min(1, vmin + raw[i] * (vmax - vmin)))
    state.canvas = mat
    if (animate) state.frame++
  }

  state.prevReset = reset
  state.prevRegen = regen
  return { image: state.canvas ? ctx.track(state.canvas.clone()) : null }
}

interface FeedbackState {
  stored: any | null
  prevReset: number
}

export const genFeedback: NodeImpl = (inputs, params, ctx) => {
  let state: FeedbackState = ctx.state.get(ctx.nodeId)
  if (!state) {
    state = { stored: null, prevReset: 0 }
    ctx.state.set(ctx.nodeId, state)
  }
  const cv = ctx.cv

  const image = inputs.image as any
  const init = inputs.init as any
  const resetRaw = inputs.reset
  const reset = typeof resetRaw === 'number' ? resetRaw : 0

  if (reset > 0.5 && state.prevReset <= 0.5) {
    if (state.stored) state.stored.delete()
    state.stored = null
  }
  state.prevReset = reset

  let out: any
  if (!state.stored) {
    if (init) {
      const f = new cv.Mat()
      init.convertTo(f, cv.CV_32F)
      out = f
    } else if (image) {
      out = new cv.Mat(image.rows, image.cols, image.channels() === 1 ? cv.CV_32F : cv.CV_32FC3, new cv.Scalar(0, 0, 0))
    } else {
      const w = Math.round(Number(params.width) || 512)
      const h = Math.round(Number(params.height) || 512)
      out = new cv.Mat(h, w, cv.CV_32F, new cv.Scalar(0))
    }
  } else {
    out = state.stored.clone()
  }

  if (image) {
    if (state.stored) state.stored.delete()
    const f = new cv.Mat()
    image.convertTo(f, image.channels() === 1 ? cv.CV_32F : cv.CV_32FC3)
    state.stored = f
  }

  return { prev: ctx.track(out) }
}

// ---------------------------------------------------------------------------
// Gray-Scott reaction-diffusion
// ---------------------------------------------------------------------------
const GS_PRESET_NAMES = ['custom', 'coral', 'mitosis', 'worms', 'spots', 'labyrinth', 'solitons', 'negatons', 'fingerprint']
const GS_PRESETS: Record<string, [number, number] | null> = {
  custom: null,
  coral: [0.0545, 0.062],
  mitosis: [0.0367, 0.0649],
  worms: [0.078, 0.061],
  spots: [0.035, 0.065],
  labyrinth: [0.03, 0.059],
  solitons: [0.03, 0.062],
  negatons: [0.046, 0.059],
  fingerprint: [0.055, 0.063],
}

interface GrayScottState {
  U: Float32Array | null
  V: Float32Array | null
  w: number
  h: number
  prevReset: number
  prevSeedTrig: number
}

function laplacianWrap(Z: Float32Array, w: number, h: number, out: Float32Array): void {
  for (let y = 0; y < h; y++) {
    const yUp = ((y - 1 + h) % h) * w
    const yDown = ((y + 1) % h) * w
    const yRow = y * w
    for (let x = 0; x < w; x++) {
      const xLeft = (x - 1 + w) % w
      const xRight = (x + 1) % w
      out[yRow + x] = Z[yUp + x] + Z[yDown + x] + Z[yRow + xLeft] + Z[yRow + xRight] - 4 * Z[yRow + x]
    }
  }
}

function toFloatResized(cv: any, ctx: any, img: any, w: number, h: number): Float32Array {
  const gray = ctx.track(toGray(cv, img))
  let src = gray
  if (gray.cols !== w || gray.rows !== h) {
    const resized = ctx.track(new cv.Mat())
    cv.resize(gray, resized, new cv.Size(w, h), 0, 0, cv.INTER_LINEAR)
    src = resized
  }
  const data = src.data as Uint8Array
  const out = new Float32Array(w * h)
  for (let i = 0; i < out.length; i++) out[i] = Math.max(0, Math.min(1, data[i] / 255))
  return out
}

export const genGrayScott: NodeImpl = (inputs, params, ctx) => {
  let state: GrayScottState = ctx.state.get(ctx.nodeId)
  if (!state) {
    state = { U: null, V: null, w: 0, h: 0, prevReset: 0, prevSeedTrig: 0 }
    ctx.state.set(ctx.nodeId, state)
  }
  const cv = ctx.cv

  const resetRaw = inputs.reset ?? params.reset
  const reset = typeof resetRaw === 'number' ? resetRaw : Number(resetRaw) || 0
  const initU = inputs.init_u as any
  const initV = inputs.init_v as any

  const w = Math.round(Number(params.width) || 256)
  const h = Math.round(Number(params.height) || 256)

  if (!state.U || (reset > 0.5 && state.prevReset <= 0.5)) {
    state.w = w
    state.h = h
    state.U = initU ? toFloatResized(cv, ctx, initU, w, h) : new Float32Array(w * h).fill(1)
    if (initV) {
      state.V = toFloatResized(cv, ctx, initV, w, h)
    } else {
      const seed = Math.round(Number(params.seed) || 42)
      const rng = makeRng(seed)
      const initMode = Number(params.init_mode) || 0
      const V = new Float32Array(w * h)
      if (initMode === 0) {
        const nSeeds = Math.max(10, Math.floor((w * h) / 1500))
        const r = Math.max(2, Math.round(Math.min(w, h) * 0.03))
        for (let s = 0; s < nSeeds; s++) {
          const px = r + Math.floor(rng() * (w - 2 * r))
          const py = r + Math.floor(rng() * (h - 2 * r))
          for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
              if ((x - px) ** 2 + (y - py) ** 2 <= r * r) V[y * w + x] = 1
            }
          }
        }
        for (let i = 0; i < V.length; i++) V[i] = Math.max(0, Math.min(1, V[i] + rng() * 0.02))
      } else {
        const rc = Math.max(2, Math.round(Math.min(w, h) * 0.05))
        const cx = Math.floor(w / 2)
        const cy = Math.floor(h / 2)
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            if ((x - cx) ** 2 + (y - cy) ** 2 <= rc * rc) V[y * w + x] = 1
          }
        }
        for (let i = 0; i < V.length; i++) V[i] = Math.max(0, Math.min(1, V[i] + rng() * 0.05))
      }
      state.V = V
    }
  }
  state.prevReset = reset

  const seedTrigRaw = inputs.seed_trig
  const seedTrig = typeof seedTrigRaw === 'number' ? seedTrigRaw : 0
  if (seedTrig > 0.5 && state.prevSeedTrig <= 0.5) {
    const sx = inputs.seed_x
    const sy = inputs.seed_y
    if (typeof sx === 'number' && typeof sy === 'number' && state.V && state.U) {
      const radiusRel = Number(params.seed_radius) || 0.05
      const cx = Math.round(sx * state.w)
      const cy = Math.round(sy * state.h)
      const r = Math.max(1, Math.round(Math.min(state.w, state.h) * radiusRel))
      for (let y = 0; y < state.h; y++) {
        for (let x = 0; x < state.w; x++) {
          if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) {
            state.V[y * state.w + x] = 1
            state.U[y * state.w + x] = 0.5
          }
        }
      }
    }
  }
  state.prevSeedTrig = seedTrig

  const presetIdx = Number(params.preset) || 0
  const presetName = GS_PRESET_NAMES[presetIdx] ?? 'custom'
  const presetFk = GS_PRESETS[presetName]
  const f = presetFk ? presetFk[0] : Number(params.f) || 0.0545
  const k = presetFk ? presetFk[1] : Number(params.k) || 0.062

  const Du = Number(params.Du) || 0.16
  const Dv = Number(params.Dv) || 0.08
  const dt = Number(params.dt) || 1.0
  const iterations = Math.round(Number(params.iterations) || 8)

  const maskImg = inputs.mask as any
  const rdMask = maskImg ? toFloatResized(cv, ctx, maskImg, state.w, state.h) : null

  const paused = !!params.pause
  if (!paused && state.U && state.V) {
    const n = state.w * state.h
    const lapU = new Float32Array(n)
    const lapV = new Float32Array(n)
    for (let iter = 0; iter < iterations; iter++) {
      laplacianWrap(state.U, state.w, state.h, lapU)
      laplacianWrap(state.V, state.w, state.h, lapV)
      const newU = new Float32Array(n)
      const newV = new Float32Array(n)
      for (let i = 0; i < n; i++) {
        const uvv = state.U[i] * state.V[i] * state.V[i]
        newU[i] = Math.max(0, Math.min(1, state.U[i] + dt * (Du * lapU[i] - uvv + f * (1 - state.U[i]))))
        newV[i] = Math.max(0, Math.min(1, state.V[i] + dt * (Dv * lapV[i] + uvv - (f + k) * state.V[i])))
      }
      if (rdMask) {
        for (let i = 0; i < n; i++) {
          newV[i] *= rdMask[i]
          newU[i] = Math.max(0, Math.min(1, newU[i] + (1 - rdMask[i]) * 0.05))
        }
      }
      state.U = newU
      state.V = newV
    }
  }

  const cmapIdx = Number(params.colormap) || 0
  const cmapNames = ['Inferno', 'Viridis', 'Jet', 'Turbo', 'Hot', null]
  const cmapName = cmapNames[cmapIdx] ?? 'Inferno'

  const v8 = ctx.track(new cv.Mat(state.h, state.w, cv.CV_8U))
  const v8Data = v8.data as Uint8Array
  for (let i = 0; i < v8Data.length; i++) v8Data[i] = Math.max(0, Math.min(255, Math.round((state.V?.[i] ?? 0) * 255)))

  let preview: any
  if (cmapName) {
    preview = ctx.track(applyColormap(cv, v8, COLORMAPS[cmapName]))
  } else {
    preview = ctx.track(new cv.Mat())
    cv.cvtColor(v8, preview, cv.COLOR_GRAY2BGR)
  }

  const uMat = ctx.track(new cv.Mat(state.h, state.w, cv.CV_32F))
  ;(uMat.data32F as Float32Array).set(state.U ?? new Float32Array(state.w * state.h))
  const vMat = ctx.track(new cv.Mat(state.h, state.w, cv.CV_32F))
  ;(vMat.data32F as Float32Array).set(state.V ?? new Float32Array(state.w * state.h))

  return { U: uMat, V: vMat, preview }
}
