import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'

interface MovieState {
  video: HTMLVideoElement
  src: string
  ready: boolean
}

function loadVideo(src: string): Promise<HTMLVideoElement> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    video.muted = true
    video.playsInline = true
    video.preload = 'auto'
    video.crossOrigin = 'anonymous'
    video.onloadedmetadata = () => resolve(video)
    video.onerror = () => reject(new Error(`Impossible de charger la vidéo: ${src}`))
    video.src = src
  })
}

function seekTo(video: HTMLVideoElement, time: number): Promise<void> {
  return new Promise((resolve) => {
    const clamped = Math.max(0, Math.min(time, video.duration || time))
    if (Math.abs(video.currentTime - clamped) < 1e-3) {
      resolve()
      return
    }
    const onSeeked = () => {
      video.removeEventListener('seeked', onSeeked)
      resolve()
    }
    video.addEventListener('seeked', onSeeked)
    video.currentTime = clamped
  })
}

export const movieFileNode: NodeDef = {
  typeId: 'input_movie',
  label: 'Movie File',
  category: 'Input',
  description: 'Charge une vidéo et en extrait une image à un instant précis. Playing avance automatiquement Frame à chaque tick.',
  inputs: [],
  outputs: [
    { id: 'main', label: 'image', color: 'image' },
    { id: 'frame', label: 'frame', color: 'scalar' },
    { id: 'total_frames', label: 'total_frames', color: 'scalar' },
  ],
  params: [
    {
      id: 'source',
      label: 'Source',
      type: 'select',
      default: '__upload__',
      options: [{ label: 'Fichier importé…', value: '__upload__' }],
    },
    { id: 'uploadedDataUrl', label: 'Fichier importé', type: 'file', default: '' },
    { id: 'fps', label: 'FPS', type: 'number', default: 25, min: 1, max: 120, step: 1 },
    { id: 'playing', label: 'Playing', type: 'boolean', default: false },
    { id: 'loop', label: 'Loop', type: 'boolean', default: false },
    { id: 'start_frame', label: 'Start Frame', type: 'number', default: 0, min: 0, max: 100000, step: 1 },
    { id: 'end_frame', label: 'End Frame (0 = fin)', type: 'number', default: 0, min: 0, max: 100000, step: 1 },
    { id: 'scrub_index', label: 'Frame', type: 'number', default: 0, min: 0, max: 100000, step: 1 },
  ],
  async process(_inputs, params, ctx) {
    const source = params.source as string
    const src = source === '__upload__' ? (params.uploadedDataUrl as string) : `${import.meta.env.BASE_URL}${source}`
    if (!src) return { main: undefined, frame: 0, total_frames: 0 }

    let state: MovieState | undefined = ctx.nodeState.get(ctx.nodeId)
    if (!state || state.src !== src) {
      const video = await loadVideo(src)
      state = { video, src, ready: true }
      ctx.nodeState.set(ctx.nodeId, state)
    }

    const fps = Number(params.fps)
    const totalFrames = Math.max(1, Math.floor((state.video.duration || 0) * fps))
    const start = Number(params.start_frame)
    const end = Number(params.end_frame) > 0 ? Number(params.end_frame) : totalFrames - 1
    const range = Math.max(1, end - start + 1)

    let idx = Number(params.scrub_index) || 0
    if (params.loop) {
      idx = start + (((idx - start) % range) + range) % range
    } else {
      idx = Math.max(start, Math.min(idx, end))
    }

    await seekTo(state.video, idx / fps)

    const canvas = document.createElement('canvas')
    canvas.width = state.video.videoWidth
    canvas.height = state.video.videoHeight
    const ctx2d = canvas.getContext('2d')!
    ctx2d.drawImage(state.video, 0, 0)
    const rgba = ctx.cv.imread(canvas)
    const bgr = trackMat(new ctx.cv.Mat())
    ctx.cv.cvtColor(rgba, bgr, ctx.cv.COLOR_RGBA2BGR)
    rgba.delete()

    return { main: bgr, frame: idx, total_frames: totalFrames }
  },
}
