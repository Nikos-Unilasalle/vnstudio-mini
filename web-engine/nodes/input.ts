import type { NodeImpl } from '../types'
import { parseColor } from '../cvUtils'
import { makeCanvas, matFromCanvas } from '../canvasCompat'

/**
 * `input.path` for `input_image` is resolved to a fetchable URL (blob:/http(s):/data:)
 * by the main thread before the graph is sent here — see resolveInputPaths in
 * shims/useVisionEngine.ts. The worker has no access to the virtual filesystem
 * that maps a saved-project path to its blob URL, only to the URL itself.
 */
type DecodedImage =
  | { type: 'bitmap'; bitmap: ImageBitmap; width: number; height: number }
  | { type: 'mat'; mat: any; width: number; height: number }

const imageCache = new Map<string, Promise<DecodedImage>>()

async function loadTiff(cv: any, buffer: ArrayBuffer): Promise<DecodedImage> {
  const { fromArrayBuffer } = await import('geotiff')
  const tiff = await fromArrayBuffer(buffer)
  const image = await tiff.getImage()
  const width = image.getWidth()
  const height = image.getHeight()
  const rgb = (await image.readRGB({ interleave: true })) as Uint8Array
  const matRgb = new cv.Mat(height, width, cv.CV_8UC3)
  matRgb.data.set(rgb)
  const matBgr = new cv.Mat()
  cv.cvtColor(matRgb, matBgr, cv.COLOR_RGB2BGR)
  matRgb.delete()
  return { type: 'mat', mat: matBgr, width, height }
}

async function loadImageData(cv: any, url: string): Promise<DecodedImage> {
  let pending = imageCache.get(url)
  if (pending) return pending

  pending = (async (): Promise<DecodedImage> => {
    if (url.startsWith('vfs:')) {
      throw new Error(`Fichier introuvable en mémoire : ${url.replace(/^vfs:\/?/, '')}`)
    }
    const baseUrl = import.meta.env.BASE_URL ?? '/'
    const fetchUrl = /^(https?:|data:|blob:|\/)/.test(url)
      ? url
      : baseUrl.replace(/\/$/, '') + '/' + url.replace(/^\//, '')

    const response = await fetch(fetchUrl)
    if (!response.ok) throw new Error(`Image introuvable : ${url}`)
    const contentType = response.headers.get('content-type') || ''
    if (contentType.includes('text/html')) {
      throw new Error(`Image introuvable : ${url}`)
    }
    const buffer = await response.arrayBuffer()
    const cleanUrl = url.split('?')[0].toLowerCase()
    const isTiff = cleanUrl.endsWith('.tif') || cleanUrl.endsWith('.tiff')

    if (isTiff) {
      try {
        return await loadTiff(cv, buffer)
      } catch (err) {
        // Fall back to createImageBitmap below
      }
    }

    const blob = new Blob([buffer])
    try {
      const bitmap = await createImageBitmap(blob)
      return { type: 'bitmap', bitmap, width: bitmap.width, height: bitmap.height }
    } catch (err) {
      // If createImageBitmap failed, try geotiff (handles TIFF files missing an extension)
      try {
        return await loadTiff(cv, buffer)
      } catch {
        throw new Error(`Format d'image non supporté ou fichier corrompu : ${url}`)
      }
    }
  })().catch((err) => {
    imageCache.delete(url)
    throw err
  })

  imageCache.set(url, pending)
  return pending
}

function drawableToBgr(cv: any, source: CanvasImageSource, width: number, height: number): any {
  const canvas = makeCanvas(width, height)
  canvas.getContext('2d')!.drawImage(source, 0, 0)
  const rgba = matFromCanvas(cv, canvas)
  const bgr = new cv.Mat()
  cv.cvtColor(rgba, bgr, cv.COLOR_RGBA2BGR)
  rgba.delete()
  return bgr
}

export const inputImage: NodeImpl = async (_inputs, params, ctx) => {
  const path = String(params.path ?? '')
  if (!path) return { main: null, width: 0, height: 0 }
  const data = await loadImageData(ctx.cv, path)
  if (data.type === 'mat') {
    const mat = ctx.track(data.mat.clone())
    return { main: mat, width: data.width, height: data.height }
  }
  const mat = ctx.track(drawableToBgr(ctx.cv, data.bitmap, data.width, data.height))
  return { main: mat, width: data.width, height: data.height }
}

/**
 * Movie/webcam frames aren't decoded here: the worker has no `<video>` element or
 * `getUserMedia`, so the main thread owns those, grabs the current frame as an
 * `ImageBitmap` before every run, and transfers it in via `ctx.frames` (see
 * MediaFrameSource in shims/useVisionEngine.ts and worker.ts).
 */
export const inputMovie: NodeImpl = (_inputs, _params, ctx) => {
  const frame = ctx.frames?.[ctx.nodeId]
  if (!frame) return { main: null, frame: 0, total_frames: 0 }
  const mat = ctx.track(drawableToBgr(ctx.cv, frame.bitmap, frame.bitmap.width, frame.bitmap.height))
  return { main: mat, frame: frame.extra?.frame ?? 0, total_frames: frame.extra?.total_frames ?? 0 }
}

export const inputWebcam: NodeImpl = (_inputs, _params, ctx) => {
  const frame = ctx.frames?.[ctx.nodeId]
  if (!frame) return { main: null, width: 0, height: 0, fps: 0 }
  const mat = ctx.track(drawableToBgr(ctx.cv, frame.bitmap, frame.bitmap.width, frame.bitmap.height))
  return { main: mat, width: frame.bitmap.width, height: frame.bitmap.height, fps: frame.extra?.fps ?? 30 }
}

export const inputSolidColor: NodeImpl = (_inputs, params, ctx) => {
  const width = Math.max(1, Number(params.width) || 640)
  const height = Math.max(1, Number(params.height) || 480)
  const mat = ctx.track(new ctx.cv.Mat(height, width, ctx.cv.CV_8UC3, parseColor(ctx.cv, String(params.color ?? '#ff0000'), [255, 0, 0])))
  return { main: mat }
}
