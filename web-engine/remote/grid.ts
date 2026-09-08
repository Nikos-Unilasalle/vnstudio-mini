/**
 * The target grid every remote source is resampled onto, and the warp that
 * puts a source raster there.
 *
 * The desktop hands this job to `odc.stac`, which hands it to GDAL. Here the
 * grid is built explicitly: a UTM raster covering the requested lon/lat box at
 * the requested metres-per-pixel, so `resolution` means what it says regardless
 * of latitude. Sources already in that UTM zone warp by a pure affine map;
 * sources in geographic coordinates go through the inverse projection.
 */
import { lonLatToUtm, utmToLonLat, utmZoneFor, type UtmZone } from '../proj'

export interface LonLatBox {
  west: number
  south: number
  east: number
  north: number
}

export interface TargetGrid {
  width: number
  height: number
  /** Metres per pixel, positive in both axes. */
  resolution: number
  zone: UtmZone
  /** UTM bounds of the grid; north/east are the far edges of the last pixel. */
  minX: number
  minY: number
  maxX: number
  maxY: number
  box: LonLatBox
}

/** The largest number of pixels a single fetch will build, to bound memory. */
const MAX_PIXELS = 4096 * 4096

/**
 * Build the UTM grid covering `box` at `resolution` m/px.
 *
 * The lon/lat box is not a rectangle in UTM, so the grid is the bounding box of
 * the projected outline — sampled along the edges, not just at the corners,
 * because meridians bow inside a zone.
 */
export function buildGrid(box: LonLatBox, resolution: number): TargetGrid {
  const zone = utmZoneFor((box.west + box.east) / 2, (box.south + box.north) / 2)
  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity
  const STEPS = 16
  for (let i = 0; i <= STEPS; i++) {
    const t = i / STEPS
    const lon = box.west + (box.east - box.west) * t
    const lat = box.south + (box.north - box.south) * t
    for (const [x, y] of [
      lonLatToUtm(lon, box.south, zone),
      lonLatToUtm(lon, box.north, zone),
      lonLatToUtm(box.west, lat, zone),
      lonLatToUtm(box.east, lat, zone),
    ]) {
      if (x < minX) minX = x
      if (x > maxX) maxX = x
      if (y < minY) minY = y
      if (y > maxY) maxY = y
    }
  }

  // Snap the origin to the resolution so repeated fetches at the same setting
  // land on the same grid and can be compared pixel for pixel.
  minX = Math.floor(minX / resolution) * resolution
  minY = Math.floor(minY / resolution) * resolution
  maxX = Math.ceil(maxX / resolution) * resolution
  maxY = Math.ceil(maxY / resolution) * resolution

  const width = Math.max(1, Math.round((maxX - minX) / resolution))
  const height = Math.max(1, Math.round((maxY - minY) / resolution))
  if (width * height > MAX_PIXELS) {
    const shrink = Math.sqrt((width * height) / MAX_PIXELS)
    const coarser = resolution * shrink
    return buildGrid(box, Math.ceil(coarser))
  }
  maxX = minX + width * resolution
  maxY = minY + height * resolution

  return { width, height, resolution, zone, minX, minY, maxX, maxY, box }
}

/** Centre of target pixel (col, row) in UTM metres. */
export function pixelCentre(grid: TargetGrid, col: number, row: number): [number, number] {
  return [grid.minX + (col + 0.5) * grid.resolution, grid.maxY - (row + 0.5) * grid.resolution]
}

/** The lon/lat bounds actually covered by the grid, which may exceed the request. */
export function gridBounds(grid: TargetGrid): LonLatBox {
  const corners: Array<[number, number]> = [
    utmToLonLat(grid.minX, grid.minY, grid.zone),
    utmToLonLat(grid.minX, grid.maxY, grid.zone),
    utmToLonLat(grid.maxX, grid.minY, grid.zone),
    utmToLonLat(grid.maxX, grid.maxY, grid.zone),
  ]
  return {
    west: Math.min(...corners.map((c) => c[0])),
    east: Math.max(...corners.map((c) => c[0])),
    south: Math.min(...corners.map((c) => c[1])),
    north: Math.max(...corners.map((c) => c[1])),
  }
}

/**
 * The rasterio-order affine transform of the grid, for the GeoRaster contract:
 * [a, b, c, d, e, f] with x = a·col + b·row + c, y = d·col + e·row + f.
 */
export function gridTransform(grid: TargetGrid): number[] {
  return [grid.resolution, 0, grid.minX, 0, -grid.resolution, grid.maxY]
}

export interface SourceRaster {
  /** Pixel values, row-major, `width × height`. */
  data: Float32Array
  width: number
  height: number
  /** Bounds of the read window in the source CRS. */
  minX: number
  minY: number
  maxX: number
  maxY: number
  /** EPSG code of the source, or null when unknown (treated as the target zone). */
  epsg: number | null
}

/**
 * Resample `source` onto `grid`.
 *
 * Pixels the source does not cover are left as `NaN` so a later composite can
 * ignore them; a categorical source uses nearest sampling so class codes are
 * never averaged into a class that does not exist.
 */
export function warpToGrid(source: SourceRaster, grid: TargetGrid, categorical: boolean): Float32Array {
  const out = new Float32Array(grid.width * grid.height).fill(NaN)
  const sw = source.width
  const sh = source.height
  if (sw < 1 || sh < 1) return out

  const spanX = source.maxX - source.minX
  const spanY = source.maxY - source.minY
  if (!(spanX > 0) || !(spanY > 0)) return out

  // Half-width of one target pixel measured in source pixels. Geographic
  // sources are metres-per-degree away from the grid, hence the conversion.
  const sourcePixelX = spanX / sw
  const sourcePixelY = spanY / sh
  const targetInSourceUnits =
    source.epsg === 4326 ? grid.resolution / 111320 : grid.resolution
  const boxRadiusX = targetInSourceUnits / (2 * sourcePixelX)
  const boxRadiusY = targetInSourceUnits / (2 * sourcePixelY)

  // Geographic sources need the inverse projection per pixel; same-zone sources
  // are a straight affine and skip it entirely.
  const geographic = source.epsg === 4326
  const sameZone = source.epsg === null || source.epsg === grid.zone.epsg

  for (let row = 0; row < grid.height; row++) {
    const y = grid.maxY - (row + 0.5) * grid.resolution
    for (let col = 0; col < grid.width; col++) {
      const x = grid.minX + (col + 0.5) * grid.resolution

      let sx: number
      let sy: number
      if (sameZone) {
        sx = x
        sy = y
      } else {
        const [lon, lat] = utmToLonLat(x, y, grid.zone)
        if (geographic) {
          sx = lon
          sy = lat
        } else {
          // Source is UTM in a different zone.
          const srcZone = { zone: (source.epsg ?? 0) % 100, north: (source.epsg ?? 0) < 32700, epsg: source.epsg ?? 0 }
          ;[sx, sy] = lonLatToUtm(lon, lat, srcZone)
        }
      }

      const fx = ((sx - source.minX) / spanX) * sw - 0.5
      const fy = ((source.maxY - sy) / spanY) * sh - 0.5
      const index = row * grid.width + col

      // Downsampling by point-sampling would alias: a 10 m band read onto a
      // 60 m grid would keep one pixel in thirty-six and throw the rest away.
      // Averaging the footprint is both what GDAL's warper does and the honest
      // answer for a reflectance value.
      if (!categorical && boxRadiusX >= 0.75) {
        const x0b = Math.max(0, Math.ceil(fx - boxRadiusX))
        const x1b = Math.min(sw - 1, Math.floor(fx + boxRadiusX))
        const y0b = Math.max(0, Math.ceil(fy - boxRadiusY))
        const y1b = Math.min(sh - 1, Math.floor(fy + boxRadiusY))
        let sum = 0
        let hits = 0
        for (let py = y0b; py <= y1b; py++) {
          const rowOffset = py * sw
          for (let px = x0b; px <= x1b; px++) {
            const v = source.data[rowOffset + px]
            if (Number.isFinite(v)) {
              sum += v
              hits++
            }
          }
        }
        if (hits > 0) {
          out[index] = sum / hits
          continue
        }
        // Nothing inside the footprint — fall through to the pointwise paths.
      }

      if (categorical) {
        const px = Math.round(fx)
        const py = Math.round(fy)
        if (px < 0 || py < 0 || px >= sw || py >= sh) continue
        out[index] = source.data[py * sw + px]
        continue
      }

      const x0 = Math.floor(fx)
      const y0 = Math.floor(fy)
      const x1 = x0 + 1
      const y1 = y0 + 1
      if (x1 < 0 || y1 < 0 || x0 >= sw || y0 >= sh) continue
      const cx0 = Math.min(sw - 1, Math.max(0, x0))
      const cx1 = Math.min(sw - 1, Math.max(0, x1))
      const cy0 = Math.min(sh - 1, Math.max(0, y0))
      const cy1 = Math.min(sh - 1, Math.max(0, y1))
      const tx = Math.min(1, Math.max(0, fx - x0))
      const ty = Math.min(1, Math.max(0, fy - y0))

      const v00 = source.data[cy0 * sw + cx0]
      const v10 = source.data[cy0 * sw + cx1]
      const v01 = source.data[cy1 * sw + cx0]
      const v11 = source.data[cy1 * sw + cx1]
      // Any NaN corner would poison the interpolation; fall back to nearest.
      if (Number.isNaN(v00) || Number.isNaN(v10) || Number.isNaN(v01) || Number.isNaN(v11)) {
        const px = Math.min(sw - 1, Math.max(0, Math.round(fx)))
        const py = Math.min(sh - 1, Math.max(0, Math.round(fy)))
        out[index] = source.data[py * sw + px]
        continue
      }
      out[index] =
        v00 * (1 - tx) * (1 - ty) + v10 * tx * (1 - ty) + v01 * (1 - tx) * ty + v11 * tx * ty
    }
  }
  return out
}

/** Combine per-scene band stacks into one, pixel by pixel, ignoring NaN. */
export function composite(
  scenes: Float32Array[],
  method: 'median' | 'mean' | 'first' | 'min' | 'max'
): Float32Array {
  if (scenes.length === 0) return new Float32Array(0)
  if (scenes.length === 1) return scenes[0]
  const n = scenes[0].length
  const out = new Float32Array(n).fill(NaN)
  const buffer = new Float64Array(scenes.length)
  for (let i = 0; i < n; i++) {
    let count = 0
    for (const scene of scenes) {
      const v = scene[i]
      if (Number.isFinite(v)) buffer[count++] = v
    }
    if (count === 0) continue
    if (method === 'first') {
      out[i] = buffer[0]
    } else if (method === 'mean') {
      let sum = 0
      for (let k = 0; k < count; k++) sum += buffer[k]
      out[i] = sum / count
    } else if (method === 'min') {
      let m = buffer[0]
      for (let k = 1; k < count; k++) if (buffer[k] < m) m = buffer[k]
      out[i] = m
    } else if (method === 'max') {
      let m = buffer[0]
      for (let k = 1; k < count; k++) if (buffer[k] > m) m = buffer[k]
      out[i] = m
    } else {
      const slice = Array.prototype.slice.call(buffer, 0, count) as number[]
      slice.sort((a, b) => a - b)
      const mid = count >> 1
      out[i] = count % 2 === 1 ? slice[mid] : (slice[mid - 1] + slice[mid]) / 2
    }
  }
  return out
}
