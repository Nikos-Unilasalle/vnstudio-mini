/**
 * The georeferenced raster the `geo_*` nodes pass between themselves.
 *
 * The desktop carries a dict around a numpy (N, H, W) array plus the metadata
 * rasterio hands back. The web build keeps the same fields, with the band stack
 * as one Float32Array per band so a raster can be sliced without copying the
 * whole cube.
 */
export interface GeoRaster {
  /** One Float32Array of width*height per band, row-major. */
  bands: Float32Array[]
  band_names: string[]
  count: number
  width: number
  height: number
  crs: string | null
  /** Affine coefficients [a, b, c, d, e, f], as rasterio orders them. */
  transform: number[] | null
  bounds: { west: number; south: number; east: number; north: number } | null
  nodata: number | null
  dtype: string
}

export function isGeoRaster(value: unknown): value is GeoRaster {
  return Boolean(value) && typeof value === 'object' && Array.isArray((value as GeoRaster).bands)
}

/** A raster sharing `source`'s georeferencing but carrying different bands. */
export function withBands(source: GeoRaster, bands: Float32Array[], names?: string[]): GeoRaster {
  return {
    ...source,
    bands,
    count: bands.length,
    band_names: names ?? bands.map((_, i) => `B${i + 1}`),
    dtype: 'float32',
  }
}

/**
 * Band `index`, counted from 1 as the node parameters do, clamped into range.
 * A raster with fewer bands than a preset expects returns its last one rather
 * than failing, which is what the desktop's `_band` helper does.
 */
export function band(raster: GeoRaster, index: number): Float32Array {
  const at = Math.min(Math.max(1, Math.round(index)), raster.count) - 1
  return raster.bands[at] ?? raster.bands[0]
}

/** (a − b) / (a + b), the shape every normalised-difference index takes. */
export function normalisedIndex(a: Float32Array, b: Float32Array, validMin: number | null = null): Float32Array {
  const out = new Float32Array(a.length)
  for (let i = 0; i < a.length; i++) {
    const x = a[i]
    const y = b[i]
    // The reference script drops near-zero reflectance so the ratio cannot blow
    // up or flip sign; without the guard the legacy path just divides.
    if (validMin !== null && (x < validMin || y < validMin)) { out[i] = NaN; continue }
    const sum = x + y
    out[i] = Math.abs(sum) < 1e-10 ? 0 : (x - y) / sum
  }
  return out
}

/** Percentile of the finite values, with numpy's linear interpolation. */
export function percentile(values: ArrayLike<number>, q: number): number {
  const finite: number[] = []
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) finite.push(values[i])
  if (finite.length === 0) return 0
  finite.sort((a, b) => a - b)
  const position = (q / 100) * (finite.length - 1)
  const low = Math.floor(position)
  const high = Math.ceil(position)
  return low === high ? finite[low] : finite[low] + (finite[high] - finite[low]) * (position - low)
}

/** A band stretched to 0-255 between two values, for display. */
export function stretchToBytes(values: Float32Array, low: number, high: number): Uint8Array {
  const out = new Uint8Array(values.length)
  const span = high - low || 1
  for (let i = 0; i < values.length; i++) {
    const v = Number.isFinite(values[i]) ? ((values[i] - low) / span) * 255 : 0
    out[i] = v < 0 ? 0 : v > 255 ? 255 : v
  }
  return out
}

/** The 2nd-98th percentile stretch the GeoTIFF reader uses for its preview. */
export function autoStretch(values: Float32Array): Uint8Array {
  const nonZero: number[] = []
  for (let i = 0; i < values.length; i++) if (values[i] !== 0 && Number.isFinite(values[i])) nonZero.push(values[i])
  if (nonZero.length === 0) return new Uint8Array(values.length)
  const low = percentile(nonZero, 2)
  const high = percentile(nonZero, 98)
  if (high === low) return new Uint8Array(values.length).fill(128)
  return stretchToBytes(values, low, high)
}

/**
 * Nearest-neighbour resample of a band onto another raster's grid.
 *
 * When both rasters carry an affine transform the mapping goes through world
 * coordinates, so two products on different grids line up properly. Without
 * one, it falls back to a plain ratio of dimensions — enough to compare a
 * reference mask that was cut from the same scene.
 */
export function resampleTo(source: GeoRaster, bandIndex: number, target: GeoRaster): Float32Array {
  const values = band(source, bandIndex)
  const out = new Float32Array(target.width * target.height)
  const st = source.transform
  const tt = target.transform

  if (st && tt && st.length >= 6 && tt.length >= 6) {
    // Invert the source affine once, then map each target centre through both.
    const [sa, sb, sc, sd, se, sf] = st
    const determinant = sa * se - sb * sd
    if (Math.abs(determinant) > 1e-18) {
      const ia = se / determinant
      const ib = -sb / determinant
      const id = -sd / determinant
      const ie = sa / determinant
      const [ta, tb, tc, td, te, tf] = tt
      for (let y = 0; y < target.height; y++) {
        for (let x = 0; x < target.width; x++) {
          const worldX = ta * (x + 0.5) + tb * (y + 0.5) + tc
          const worldY = td * (x + 0.5) + te * (y + 0.5) + tf
          const dx = worldX - sc
          const dy = worldY - sf
          const sx = Math.floor(ia * dx + ib * dy)
          const sy = Math.floor(id * dx + ie * dy)
          out[y * target.width + x] =
            sx >= 0 && sy >= 0 && sx < source.width && sy < source.height ? values[sy * source.width + sx] : 0
        }
      }
      return out
    }
  }

  for (let y = 0; y < target.height; y++) {
    const sy = Math.min(source.height - 1, Math.floor((y * source.height) / target.height))
    for (let x = 0; x < target.width; x++) {
      const sx = Math.min(source.width - 1, Math.floor((x * source.width) / target.width))
      out[y * target.width + x] = values[sy * source.width + sx]
    }
  }
  return out
}

/** Mulberry32, so a seeded Monte-Carlo run repeats exactly. */
export function seededRandom(seed: number): () => number {
  let state = (seed >>> 0) || 1
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Standard normal draws by the Box-Muller transform.
 *
 * numpy's Generator uses the ziggurat, so the individual numbers differ from a
 * desktop run even at the same seed; the distribution is the same, which is
 * what a Monte-Carlo ensemble is actually asking for.
 */
export function gaussianField(length: number, random: () => number): Float32Array {
  const out = new Float32Array(length)
  for (let i = 0; i < length; i += 2) {
    const u = Math.max(random(), 1e-12)
    const v = random()
    const radius = Math.sqrt(-2 * Math.log(u))
    const angle = 2 * Math.PI * v
    out[i] = radius * Math.cos(angle)
    if (i + 1 < length) out[i + 1] = radius * Math.sin(angle)
  }
  return out
}
