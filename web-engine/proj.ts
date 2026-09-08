/**
 * The map projections the remote-imagery nodes need.
 *
 * The desktop leans on GDAL/pyproj for this. In the browser there is no such
 * library within reach, but the two projections that actually matter here have
 * closed forms: transverse Mercator (UTM), which is the native grid of every
 * Sentinel-2 tile, and spherical Mercator, which is the grid of every XYZ tile
 * server. Both are implemented from Snyder's series, accurate to well under a
 * millimetre inside a zone — far finer than the 10 m pixels they position.
 */

const A = 6378137.0                    // WGS84 semi-major axis (m)
const F = 1 / 298.257223563            // flattening
const E2 = F * (2 - F)                 // first eccentricity squared
const EP2 = E2 / (1 - E2)              // second eccentricity squared
const K0 = 0.9996                      // UTM scale factor on the central meridian
const FALSE_EASTING = 500000.0
const FALSE_NORTHING = 10000000.0      // southern hemisphere only

export interface UtmZone {
  zone: number
  north: boolean
  epsg: number
}

/** The UTM zone a lon/lat falls in — the grid the raster will be built on. */
export function utmZoneFor(lon: number, lat: number): UtmZone {
  const zone = Math.min(60, Math.max(1, Math.floor((lon + 180) / 6) + 1))
  const north = lat >= 0
  return { zone, north, epsg: (north ? 32600 : 32700) + zone }
}

/** Parse the EPSG code out of whatever form a GeoTIFF or STAC item reports. */
export function utmZoneFromEpsg(epsg: number): UtmZone | null {
  if (epsg >= 32601 && epsg <= 32660) return { zone: epsg - 32600, north: true, epsg }
  if (epsg >= 32701 && epsg <= 32760) return { zone: epsg - 32700, north: false, epsg }
  return null
}

function centralMeridian(zone: number): number {
  return (zone - 1) * 6 - 180 + 3
}

/** Meridional arc length from the equator to `lat` (radians). */
function meridianArc(lat: number): number {
  const n = (1 - Math.sqrt(1 - E2)) / (1 + Math.sqrt(1 - E2))
  const n2 = n * n
  const n3 = n2 * n
  const n4 = n3 * n
  // Series in n converges much faster than the classic series in e².
  return (
    (A / (1 + n)) *
    ((1 + n2 / 4 + n4 / 64) * lat -
      1.5 * (n - n3 / 8) * Math.sin(2 * lat) +
      (15 / 16) * (n2 - n4 / 4) * Math.sin(4 * lat) -
      (35 / 48) * n3 * Math.sin(6 * lat) +
      (315 / 512) * n4 * Math.sin(8 * lat))
  )
}

/** lon/lat (degrees) → UTM easting/northing (metres) in the given zone. */
export function lonLatToUtm(lon: number, lat: number, zone: UtmZone): [number, number] {
  const phi = (lat * Math.PI) / 180
  const lam = (lon * Math.PI) / 180
  const lam0 = (centralMeridian(zone.zone) * Math.PI) / 180

  const sinPhi = Math.sin(phi)
  const cosPhi = Math.cos(phi)
  const tanPhi = Math.tan(phi)

  const N = A / Math.sqrt(1 - E2 * sinPhi * sinPhi)
  const T = tanPhi * tanPhi
  const C = EP2 * cosPhi * cosPhi
  let dl = lam - lam0
  // Keep the longitude difference in (-π, π] so a zone straddling the antimeridian works.
  while (dl > Math.PI) dl -= 2 * Math.PI
  while (dl < -Math.PI) dl += 2 * Math.PI
  const Aq = dl * cosPhi
  const A2 = Aq * Aq
  const M = meridianArc(phi)

  const easting =
    K0 * N * (Aq + ((1 - T + C) * A2 * Aq) / 6 + ((5 - 18 * T + T * T + 72 * C - 58 * EP2) * A2 * A2 * Aq) / 120) +
    FALSE_EASTING

  let northing =
    K0 *
    (M +
      N *
        tanPhi *
        (A2 / 2 +
          ((5 - T + 9 * C + 4 * C * C) * A2 * A2) / 24 +
          ((61 - 58 * T + T * T + 600 * C - 330 * EP2) * A2 * A2 * A2) / 720))

  if (!zone.north) northing += FALSE_NORTHING
  return [easting, northing]
}

/** UTM easting/northing (metres) → lon/lat (degrees). */
export function utmToLonLat(easting: number, northing: number, zone: UtmZone): [number, number] {
  const x = easting - FALSE_EASTING
  const y = zone.north ? northing : northing - FALSE_NORTHING

  const e1 = (1 - Math.sqrt(1 - E2)) / (1 + Math.sqrt(1 - E2))
  const M = y / K0
  const mu = M / (A * (1 - E2 / 4 - (3 * E2 * E2) / 64 - (5 * E2 * E2 * E2) / 256))

  const phi1 =
    mu +
    ((3 * e1) / 2 - (27 * e1 ** 3) / 32) * Math.sin(2 * mu) +
    ((21 * e1 * e1) / 16 - (55 * e1 ** 4) / 32) * Math.sin(4 * mu) +
    ((151 * e1 ** 3) / 96) * Math.sin(6 * mu) +
    ((1097 * e1 ** 4) / 512) * Math.sin(8 * mu)

  const sinPhi1 = Math.sin(phi1)
  const cosPhi1 = Math.cos(phi1)
  const tanPhi1 = Math.tan(phi1)

  const C1 = EP2 * cosPhi1 * cosPhi1
  const T1 = tanPhi1 * tanPhi1
  const N1 = A / Math.sqrt(1 - E2 * sinPhi1 * sinPhi1)
  const R1 = (A * (1 - E2)) / Math.pow(1 - E2 * sinPhi1 * sinPhi1, 1.5)
  const D = x / (N1 * K0)
  const D2 = D * D

  const phi =
    phi1 -
    ((N1 * tanPhi1) / R1) *
      (D2 / 2 -
        ((5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * EP2) * D2 * D2) / 24 +
        ((61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 - 252 * EP2 - 3 * C1 * C1) * D2 * D2 * D2) / 720)

  const lam =
    (D -
      ((1 + 2 * T1 + C1) * D2 * D) / 6 +
      ((5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * EP2 + 24 * T1 * T1) * D2 * D2 * D) / 120) /
    cosPhi1

  const lam0 = (centralMeridian(zone.zone) * Math.PI) / 180
  return [((lam0 + lam) * 180) / Math.PI, (phi * 180) / Math.PI]
}

// ── Spherical Mercator, for XYZ tile servers ─────────────────────────────────

export const WEB_MERCATOR_EXTENT = 20037508.342789244

/** lon/lat → fractional tile coordinates at zoom `z` (origin top-left). */
export function lonLatToTile(lon: number, lat: number, z: number): [number, number] {
  const n = Math.pow(2, z)
  const x = ((lon + 180) / 360) * n
  const clamped = Math.max(-85.05112878, Math.min(85.05112878, lat))
  const rad = (clamped * Math.PI) / 180
  const y = ((1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2) * n
  return [x, y]
}

/** Fractional tile coordinates at zoom `z` → lon/lat. */
export function tileToLonLat(x: number, y: number, z: number): [number, number] {
  const n = Math.pow(2, z)
  const lon = (x / n) * 360 - 180
  const lat = (Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / n))) * 180) / Math.PI
  return [lon, lat]
}

/**
 * The zoom level whose pixels are closest to `metresPerPixel` at this latitude.
 * Web Mercator pixels shrink towards the poles, hence the cos(lat) factor.
 */
export function zoomForResolution(metresPerPixel: number, lat: number, tileSize = 256): number {
  const groundWidth = (2 * Math.PI * A * Math.cos((lat * Math.PI) / 180)) / tileSize
  const z = Math.log2(groundWidth / Math.max(0.01, metresPerPixel))
  return Math.max(0, Math.min(21, Math.round(z)))
}
