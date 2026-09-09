/**
 * Copernicus Data Space Ecosystem — Sentinel Hub Process API.
 *
 * The one backend that needs credentials. They are never a graph parameter: the
 * node reads them from the local key vault (`.vnstudio/secrets.json`, which the
 * API-keys panel writes to browser storage and which the graph serialiser never
 * touches), so sharing a `.vn` file never ships someone's client secret.
 *
 * Sentinel Hub returns an already-projected, already-resampled GeoTIFF for the
 * requested CRS and pixel grid, so unlike the STAC path there is nothing to
 * warp here — the response *is* the target raster.
 */
import { readTextFile } from '../textFiles'
import { request } from './request'
import type { TargetGrid } from './grid'

const TOKEN_URL =
  'https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token'
const PROCESS_URL = 'https://sh.dataspace.copernicus.eu/api/v1/process'
const SECRETS_PATH = '.vnstudio/secrets.json'

export interface Credentials {
  clientId: string
  clientSecret: string
}

/** Credentials from the local vault, or null when the user has not set them. */
export function readCredentials(): Credentials | null {
  const raw = readTextFile(SECRETS_PATH)
  if (!raw) return null
  try {
    const parsed = JSON.parse(raw) as Record<string, string>
    const clientId = (parsed.copernicus_client_id ?? '').trim()
    const clientSecret = (parsed.copernicus_client_secret ?? '').trim()
    if (!clientId || !clientSecret) return null
    return { clientId, clientSecret }
  } catch {
    return null
  }
}

let token: { value: string; expiresAt: number } | null = null

async function accessToken(credentials: Credentials): Promise<string> {
  if (token && token.expiresAt - 60_000 > Date.now()) return token.value
  const body = new URLSearchParams({
    grant_type: 'client_credentials',
    client_id: credentials.clientId,
    client_secret: credentials.clientSecret,
  })
  const response = await request(TOKEN_URL, {
    method: 'POST',
    headers: { 'content-type': 'application/x-www-form-urlencoded' },
    body,
  })
  if (response.status === 401) {
    throw new Error('CDSE : identifiants refusés — vérifie Client ID et Secret dans le panneau de clés')
  }
  if (!response.ok) throw new Error(`CDSE : jeton refusé (${response.status})`)
  const payload = (await response.json()) as { access_token?: string; expires_in?: number }
  if (!payload.access_token) throw new Error('CDSE : réponse sans jeton')
  token = {
    value: payload.access_token,
    expiresAt: Date.now() + (payload.expires_in ?? 600) * 1000,
  }
  return token.value
}

/** Forget the cached token — used when a request comes back unauthorised. */
export function resetToken(): void {
  token = null
}

/**
 * The evalscript telling Sentinel Hub which bands to return and in what form.
 *
 * `SAMPLE_TYPE: FLOAT32` keeps physical units: reflectance stays in 0–1 rather
 * than being stretched to bytes, so downstream indices mean what they should.
 */
function evalscript(bands: string[], toDecibels: boolean, units: string | null): string {
  const outputs = bands
    .map((band) =>
      toDecibels
        ? `(sample.${band} > 0 ? 10 * Math.log(sample.${band}) / Math.LN10 : -50)`
        : `sample.${band}`
    )
    .join(', ')
  const inputUnits = units ? `, units: "${units}"` : ''
  return `//VERSION=3
function setup() {
  return {
    input: [{ bands: [${bands.map((b) => `"${b}"`).join(', ')}]${inputUnits} }],
    output: { bands: ${bands.length}, sampleType: "FLOAT32" }
  }
}
function evaluatePixel(sample) {
  return [${outputs}]
}`
}

export interface CdseOptions {
  collectionId: string
  bands: string[]
  grid: TargetGrid
  dateRange: [string, string]
  cloudMax: number | null
  /** `leastCC` picks the clearest pixel over the window; `mostRecent` the newest. */
  mosaickingOrder: 'leastCC' | 'mostRecent'
  toDecibels: boolean
  units: string | null
  /** Selects between GLO-30 and GLO-90, which share the `dem` collection type. */
  demInstance?: string | null
}

/** One Process API call, returning one Float32Array per requested band. */
export async function fetchCdse(options: CdseOptions): Promise<Float32Array[]> {
  const credentials = readCredentials()
  if (!credentials) {
    throw new Error(
      'CDSE : aucun identifiant enregistré — ouvre le panneau de clés (roue dentée) ' +
        'et renseigne Client ID et Client Secret Copernicus'
    )
  }
  const bearer = await accessToken(credentials)
  const { grid } = options

  const dataFilter: Record<string, unknown> = {
    timeRange: { from: `${options.dateRange[0]}T00:00:00Z`, to: `${options.dateRange[1]}T23:59:59Z` },
    mosaickingOrder: options.mosaickingOrder,
  }
  if (options.cloudMax !== null) dataFilter.maxCloudCoverage = options.cloudMax
  if (options.demInstance) dataFilter.demInstance = options.demInstance

  const payload = {
    input: {
      bounds: {
        bbox: [grid.minX, grid.minY, grid.maxX, grid.maxY],
        properties: { crs: `http://www.opengis.net/def/crs/EPSG/0/${grid.zone.epsg}` },
      },
      data: [{ type: options.collectionId, dataFilter }],
    },
    output: {
      width: grid.width,
      height: grid.height,
      responses: [{ identifier: 'default', format: { type: 'image/tiff' } }],
    },
    evalscript: evalscript(options.bands, options.toDecibels, options.units),
  }

  const response = await request(PROCESS_URL, {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: `Bearer ${bearer}`, accept: 'image/tiff' },
    body: JSON.stringify(payload),
  })
  if (response.status === 401) {
    resetToken()
    throw new Error('CDSE : jeton rejeté — réessaie, ou vérifie les identifiants')
  }
  if (!response.ok) {
    const detail = await response.text().catch(() => '')
    throw new Error(`CDSE : requête refusée (${response.status}) ${detail.slice(0, 300)}`)
  }

  const buffer = await response.arrayBuffer()
  const { fromArrayBuffer } = await import('geotiff')
  const tiff = await fromArrayBuffer(buffer)
  const image = await tiff.getImage()
  const rasters = (await image.readRasters({ interleave: false })) as unknown as {
    length: number
    [index: number]: ArrayLike<number>
  }

  const out: Float32Array[] = []
  for (let b = 0; b < options.bands.length; b++) {
    const source = rasters[b]
    if (!source) break
    const band = new Float32Array(grid.width * grid.height)
    for (let i = 0; i < band.length && i < source.length; i++) band[i] = source[i]
    out.push(band)
  }
  if (out.length === 0) throw new Error('CDSE : réponse vide')
  return out
}
