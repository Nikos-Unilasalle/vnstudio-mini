import type { NodeImpl } from '../types'
import { parseColor } from '../cvUtils'
import { resolveMediaUrl } from '../../shims/vfs'

const imageCache = new Map<string, Promise<HTMLImageElement>>()

function loadImage(src: string): Promise<HTMLImageElement> {
  let pending = imageCache.get(src)
  if (pending) return pending
  pending = new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => resolve(img)
    img.onerror = () => reject(new Error(`Image introuvable : ${src}`))
    img.src = src
  })
  imageCache.set(src, pending)
  return pending
}

function drawableToBgr(cv: any, source: CanvasImageSource, width: number, height: number): any {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  canvas.getContext('2d')!.drawImage(source, 0, 0)
  const rgba = cv.imread(canvas)
  const bgr = new cv.Mat()
  cv.cvtColor(rgba, bgr, cv.COLOR_RGBA2BGR)
  rgba.delete()
  return bgr
}

export const inputImage: NodeImpl = async (_inputs, params, ctx) => {
  const path = String(params.path ?? '')
  if (!path) return { main: null, width: 0, height: 0 }
  const img = await loadImage(resolveMediaUrl(path))
  const mat = ctx.track(drawableToBgr(ctx.cv, img, img.naturalWidth, img.naturalHeight))
  return { main: mat, width: img.naturalWidth, height: img.naturalHeight }
}

interface MovieState {
  video: HTMLVideoElement
  src: string
}

function loadVideo(src: string): Promise<HTMLVideoElement> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    video.muted = true
    video.playsInline = true
    video.preload = 'auto'
    video.crossOrigin = 'anonymous'
    video.onloadedmetadata = () => resolve(video)
    video.onerror = () => reject(new Error(`Vidéo introuvable : ${src}`))
    video.src = src
  })
}

function seek(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve) => {
    const target = Math.max(0, Math.min(time, Math.max(0, (video.duration || 0) - 0.01)))
    if (Math.abs(video.currentTime - target) < 1e-3) {
      resolve()
      return
    }
    const onSeeked = () => {
      video.removeEventListener('seeked', onSeeked)
      resolve()
    }
    video.addEventListener('seeked', onSeeked)
    video.currentTime = target
  })
}

/** Desktop reads fps from the container; in the browser it is not exposed, so assume 25. */
const ASSUMED_FPS = 25

export const inputMovie: NodeImpl = async (_inputs, params, ctx) => {
  const path = String(params.path ?? '')
  if (!path) return { main: null, frame: 0, total_frames: 0 }

  const src = resolveMediaUrl(path)
  let state: MovieState | undefined = ctx.state.get(ctx.nodeId)
  if (!state || state.src !== src) {
    state = { video: await loadVideo(src), src }
    ctx.state.set(ctx.nodeId, state)
  }

  const totalFrames = Math.max(1, Math.floor((state.video.duration || 0) * ASSUMED_FPS))
  const start = Math.max(0, Number(params.start_frame) || 0)
  const rawEnd = Number(params.end_frame) || 0
  const end = rawEnd > 0 ? Math.min(rawEnd, totalFrames - 1) : totalFrames - 1
  const span = Math.max(1, end - start + 1)

  let index = Number(params.scrub_index) || 0
  if (params.loop) index = start + (((index - start) % span) + span) % span
  else index = Math.max(start, Math.min(index, end))

  await seek(state.video, index / ASSUMED_FPS)
  const mat = ctx.track(drawableToBgr(ctx.cv, state.video, state.video.videoWidth, state.video.videoHeight))
  return { main: mat, frame: index, total_frames: totalFrames }
}

interface WebcamState {
  video: HTMLVideoElement
  stream: MediaStream
}

export const inputWebcam: NodeImpl = async (_inputs, _params, ctx) => {
  let state: WebcamState | undefined = ctx.state.get(ctx.nodeId)
  if (!state) {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true })
    const video = document.createElement('video')
    video.srcObject = stream
    video.muted = true
    video.playsInline = true
    await video.play()
    state = { video, stream }
    ctx.state.set(ctx.nodeId, state)
  }
  const { video } = state
  if (!video.videoWidth) return { main: null, width: 0, height: 0, fps: 0 }
  const mat = ctx.track(drawableToBgr(ctx.cv, video, video.videoWidth, video.videoHeight))
  return { main: mat, width: video.videoWidth, height: video.videoHeight, fps: 30 }
}

export const inputSolidColor: NodeImpl = (_inputs, params, ctx) => {
  const width = Math.max(1, Number(params.width) || 640)
  const height = Math.max(1, Number(params.height) || 480)
  const mat = ctx.track(new ctx.cv.Mat(height, width, ctx.cv.CV_8UC3, parseColor(ctx.cv, String(params.color ?? '#ff0000'), [255, 0, 0])))
  return { main: mat }
}
