/**
 * XYZ tile servers — Google Satellite, OpenStreetMap, Carto.
 *
 * These have no STAC catalogue and no georeferenced files: they hand out 256 px
 * PNG/JPEG tiles on a spherical Mercator grid. The mosaic is stitched on an
 * OffscreenCanvas, then sampled onto the same UTM grid every other source lands
 * on, so a basemap can be stacked with satellite bands without special-casing.
 */
import { lonLatToTile, utmToLonLat, zoomForResolution } from '../proj'
import type { TargetGrid } from './grid'
import { request } from './request'

const TILE = 256
/** Above this many tiles the request is refused rather than hammering a server. */
const MAX_TILES = 256

export interface BasemapResult {
  /** One Float32Array per channel, 0–255, on the target grid. */
  bands: Float32Array[]
  bandNames: string[]
  zoom: number
  tiles: number
  /** Tiles the server did not return; the mosaic keeps their area black. */
  missing: number
}

async function loadTile(url: string): Promise<ImageBitmap | null> {
  const response = await request(url)
  if (!response.ok) return null
  try {
    return await createImageBitmap(await response.blob())
  } catch {
    return null
  }
}

/** Fetch and stitch the tiles covering `grid`, then sample them onto it. */
export async function fetchBasemap(
  urlTemplate: string,
  grid: TargetGrid,
  onProgress?: (fraction: number, message: string) => void
): Promise<BasemapResult> {
  const box = grid.box
  const lat = (box.south + box.north) / 2
  const zoom = zoomForResolution(grid.resolution, lat, TILE)

  const [tx0f, ty0f] = lonLatToTile(box.west, box.north, zoom)
  const [tx1f, ty1f] = lonLatToTile(box.east, box.south, zoom)
  const span = Math.pow(2, zoom)
  const tx0 = Math.max(0, Math.floor(Math.min(tx0f, tx1f)))
  const tx1 = Math.min(span - 1, Math.floor(Math.max(tx0f, tx1f)))
  const ty0 = Math.max(0, Math.floor(Math.min(ty0f, ty1f)))
  const ty1 = Math.min(span - 1, Math.floor(Math.max(ty0f, ty1f)))

  const nx = tx1 - tx0 + 1
  const ny = ty1 - ty0 + 1
  if (nx * ny > MAX_TILES) {
    throw new Error(
      `emprise trop large pour ce fond de carte : ${nx * ny} tuiles au zoom ${zoom} ` +
        `(maximum ${MAX_TILES}) — augmente la résolution en m/px`
    )
  }

  const canvas = new OffscreenCanvas(nx * TILE, ny * TILE)
  const context = canvas.getContext('2d')
  if (!context) throw new Error('OffscreenCanvas 2D indisponible')
  context.fillStyle = '#000'
  context.fillRect(0, 0, canvas.width, canvas.height)

  let done = 0
  let missing = 0
  const total = nx * ny
  for (let ty = ty0; ty <= ty1; ty++) {
    for (let tx = tx0; tx <= tx1; tx++) {
      const url = urlTemplate
        .replace('{x}', String(tx))
        .replace('{y}', String(ty))
        .replace('{z}', String(zoom))
      let bitmap: ImageBitmap | null = null
      try {
        bitmap = await loadTile(url)
      } catch (error) {
        // The first tile proves whether the server is reachable at all; after
        // that a gap is just a gap, and a mosaic with a hole beats no mosaic.
        if (done === 0) throw error
      }
      done += 1
      onProgress?.(done / total, `tuile ${done}/${total} (zoom ${zoom})`)
      if (!bitmap) {
        missing += 1
        continue
      }
      context.drawImage(bitmap, (tx - tx0) * TILE, (ty - ty0) * TILE, TILE, TILE)
      bitmap.close()
    }
  }

  const mosaic = context.getImageData(0, 0, canvas.width, canvas.height)
  const pixels = mosaic.data
  const mw = canvas.width
  const mh = canvas.height

  const channels = [new Float32Array(grid.width * grid.height), new Float32Array(grid.width * grid.height), new Float32Array(grid.width * grid.height)]
  for (let row = 0; row < grid.height; row++) {
    const y = grid.maxY - (row + 0.5) * grid.resolution
    for (let col = 0; col < grid.width; col++) {
      const x = grid.minX + (col + 0.5) * grid.resolution
      const [lon, latP] = utmToLonLat(x, y, grid.zone)
      const [gx, gy] = lonLatToTile(lon, latP, zoom)
      const px = Math.round((gx - tx0) * TILE - 0.5)
      const py = Math.round((gy - ty0) * TILE - 0.5)
      const index = row * grid.width + col
      if (px < 0 || py < 0 || px >= mw || py >= mh) continue
      const offset = (py * mw + px) * 4
      channels[0][index] = pixels[offset]
      channels[1][index] = pixels[offset + 1]
      channels[2][index] = pixels[offset + 2]
    }
  }

  if (missing === total) {
    throw new Error(`aucune tuile reçue de ${new URL(urlTemplate.replace(/\{[xyz]\}/g, '0')).host}`)
  }
  return { bands: channels, bandNames: ['R', 'G', 'B'], zoom, tiles: total, missing }
}
