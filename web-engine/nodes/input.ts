import type { NodeImpl } from '../types'
import { parseColor } from '../cvUtils'
import { makeCanvas, matFromCanvas } from '../canvasCompat'

/**
 * `input.path` for `input_image` is resolved to a fetchable URL (blob:/http(s):/data:)
 * by the main thread before the graph is sent here — see resolveInputPaths in
 * shims/useVisionEngine.ts. The worker has no access to the virtual filesystem
 * that maps a saved-project path to its blob URL, only to the URL itself.
 */
const imageCache = new Map<string, Promise<ImageBitmap>>()

function loadImage(url: string): Promise<ImageBitmap> {
  let pending = imageCache.get(url)
  if (pending) return pending
  pending = fetch(url)
    .then((response) => {
      if (!response.ok) throw new Error(`Image introuvable : ${url}`)
      return response.blob()
    })
    .then((blob) => createImageBitmap(blob))
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
  const bitmap = await loadImage(path)
  const mat = ctx.track(drawableToBgr(ctx.cv, bitmap, bitmap.width, bitmap.height))
  return { main: mat, width: bitmap.width, height: bitmap.height }
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
