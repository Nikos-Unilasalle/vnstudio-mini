/**
 * Small colormap LUTs, for OpenCV builds that omit `cv.applyColorMap`
 * (confirmed absent from the opencv-wasm@4.3.0-10 imgproc bindings this
 * project loads — see opencv.ts).  Returns [r, g, b] for v in [0, 255].
 */

/** Standard "jet" formula — piecewise-linear ramps in each channel. */
export function jetColor(v: number): [number, number, number] {
  const t = Math.max(0, Math.min(255, v)) / 255
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 3)))
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 2)))
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 1)))
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]
}

// A handful of control points sampled from matplotlib's viridis, linearly
// interpolated — a close visual approximation without embedding the full 256-entry table.
const VIRIDIS_STOPS: [number, number, number][] = [
  [68, 1, 84],
  [72, 40, 120],
  [62, 74, 137],
  [49, 104, 142],
  [38, 130, 142],
  [31, 158, 137],
  [53, 183, 121],
  [109, 205, 89],
  [180, 222, 44],
  [253, 231, 37],
]

export function viridisColor(v: number): [number, number, number] {
  const t = (Math.max(0, Math.min(255, v)) / 255) * (VIRIDIS_STOPS.length - 1)
  const i = Math.min(VIRIDIS_STOPS.length - 2, Math.floor(t))
  const f = t - i
  const [r0, g0, b0] = VIRIDIS_STOPS[i]
  const [r1, g1, b1] = VIRIDIS_STOPS[i + 1]
  return [Math.round(r0 + (r1 - r0) * f), Math.round(g0 + (g1 - g0) * f), Math.round(b0 + (b1 - b0) * f)]
}

function stopsColor(stops: [number, number, number][], v: number): [number, number, number] {
  const t = (Math.max(0, Math.min(255, v)) / 255) * (stops.length - 1)
  const i = Math.min(stops.length - 2, Math.floor(t))
  const f = t - i
  const [r0, g0, b0] = stops[i]
  const [r1, g1, b1] = stops[i + 1]
  return [Math.round(r0 + (r1 - r0) * f), Math.round(g0 + (g1 - g0) * f), Math.round(b0 + (b1 - b0) * f)]
}

const PLASMA_STOPS: [number, number, number][] = [
  [13, 8, 135], [84, 2, 163], [139, 10, 165], [185, 50, 137],
  [219, 92, 104], [244, 136, 73], [254, 188, 43], [240, 249, 33],
]
export function plasmaColor(v: number): [number, number, number] {
  return stopsColor(PLASMA_STOPS, v)
}

const INFERNO_STOPS: [number, number, number][] = [
  [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
  [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164],
]
export function infernoColor(v: number): [number, number, number] {
  return stopsColor(INFERNO_STOPS, v)
}

const MAGMA_STOPS: [number, number, number][] = [
  [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
  [212, 72, 66], [245, 125, 21], [252, 172, 108], [252, 253, 191],
]
export function magmaColor(v: number): [number, number, number] {
  return stopsColor(MAGMA_STOPS, v)
}

const TURBO_STOPS: [number, number, number][] = [
  [48, 18, 59], [70, 107, 227], [40, 187, 233], [66, 220, 157],
  [172, 240, 60], [246, 202, 43], [231, 106, 27], [151, 27, 6],
]
export function turboColor(v: number): [number, number, number] {
  return stopsColor(TURBO_STOPS, v)
}

export function hotColor(v: number): [number, number, number] {
  const t = Math.max(0, Math.min(255, v)) / 255
  const r = Math.min(1, t / 0.365)
  const g = Math.min(1, Math.max(0, (t - 0.365) / 0.38))
  const b = Math.min(1, Math.max(0, (t - 0.745) / 0.255))
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)]
}

export function coolColor(v: number): [number, number, number] {
  const t = Math.max(0, Math.min(255, v)) / 255
  return [Math.round(t * 255), Math.round((1 - t) * 255), 255]
}

const OCEAN_STOPS: [number, number, number][] = [
  [0, 0, 0], [0, 0, 128], [0, 90, 100], [0, 160, 90], [130, 200, 60], [255, 255, 255],
]
export function oceanColor(v: number): [number, number, number] {
  return stopsColor(OCEAN_STOPS, v)
}

const RAINBOW_STOPS: [number, number, number][] = [
  [128, 0, 128], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0],
]
export function rainbowColor(v: number): [number, number, number] {
  return stopsColor(RAINBOW_STOPS, v)
}

const PARULA_STOPS: [number, number, number][] = [
  [53, 42, 135], [15, 92, 175], [20, 129, 168], [42, 155, 137],
  [104, 172, 99], [183, 177, 71], [253, 190, 61], [249, 251, 14],
]
export function parulaColor(v: number): [number, number, number] {
  return stopsColor(PARULA_STOPS, v)
}

const CIVIDIS_STOPS: [number, number, number][] = [
  [0, 32, 76], [0, 42, 102], [64, 60, 92], [110, 82, 95], [147, 105, 97], [204, 158, 90], [255, 217, 82],
]
export function cividisColor(v: number): [number, number, number] {
  return stopsColor(CIVIDIS_STOPS, v)
}

export const COLORMAPS: Record<string, (v: number) => [number, number, number]> = {
  Viridis: viridisColor,
  Plasma: plasmaColor,
  Inferno: infernoColor,
  Magma: magmaColor,
  Turbo: turboColor,
  Jet: jetColor,
  Hot: hotColor,
  Cool: coolColor,
  Parula: parulaColor,
  Cividis: cividisColor,
  Rainbow: rainbowColor,
  Ocean: oceanColor,
}

/** Applies a colormap function to a single-channel byte Mat, producing a new BGR Mat. */
export function applyColormap(cv: any, gray: any, colorFn: (v: number) => [number, number, number]): any {
  const lut: [number, number, number][] = []
  for (let i = 0; i < 256; i++) lut[i] = colorFn(i)

  const out = new cv.Mat(gray.rows, gray.cols, cv.CV_8UC3)
  const src = gray.data as Uint8Array
  const dst = out.data as Uint8Array
  for (let i = 0, px = 0; i < src.length; i++, px += 3) {
    const [r, g, b] = lut[src[i]]
    dst[px] = b
    dst[px + 1] = g
    dst[px + 2] = r
  }
  return out
}
