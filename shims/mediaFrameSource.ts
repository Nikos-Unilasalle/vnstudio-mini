/**
 * Owns `<video>` elements and the webcam stream on the main thread, on behalf
 * of `input_movie` / `input_webcam` nodes.
 *
 * The worker that runs the graph has no DOM — no `<video>`, no
 * `getUserMedia` — so it can't decode video itself. Before every run, the
 * main thread grabs the *current* frame of each such node as a transferable
 * `ImageBitmap` and hands it to the worker via `ctx.frames` (see
 * web-engine/nodes/input.ts and worker.ts). This module is what produces
 * those bitmaps.
 */
import { resolveMediaUrl } from './vfs'
import type { CapturedFrame } from '../web-engine/types'

/** Desktop reads fps from the container; in the browser it is not exposed, so assume 25. */
const ASSUMED_FPS = 25

/** Longest a seek may hold up a run; past it the frame is taken where the video is. */
const SEEK_TIMEOUT_MS = 1500

/**
 * `video` is null while the file loads or the camera opens. Both happen off the
 * run: a run never waits on them, it just gets no frame yet — waiting would
 * freeze the whole graph on a camera prompt the student has not answered.
 */
interface MovieState {
  video: HTMLVideoElement | null
  src: string
  closed: boolean
}

interface WebcamState {
  video: HTMLVideoElement | null
  stream: MediaStream | null
  closed: boolean
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
    const done = () => {
      video.removeEventListener('seeked', done)
      clearTimeout(timer)
      resolve()
    }
    const timer = setTimeout(done, SEEK_TIMEOUT_MS)
    video.addEventListener('seeked', done)
    video.currentTime = target
  })
}

export class MediaFrameSource {
  private readonly movies = new Map<string, MovieState>()
  private readonly webcams = new Map<string, WebcamState>()

  /** `onReady` asks for a run once a video or camera that was opening can deliver frames. */
  constructor(private readonly onReady?: () => void) {}

  /** Drops state for nodes that no longer exist, stopping their video/webcam. */
  pruneState(liveIds: Set<string>): void {
    for (const [id, state] of this.movies) {
      if (!liveIds.has(id)) {
        this.closeMovie(state)
        this.movies.delete(id)
      }
    }
    for (const [id, state] of this.webcams) {
      if (!liveIds.has(id)) {
        state.closed = true
        for (const track of state.stream?.getTracks() ?? []) track.stop()
        this.webcams.delete(id)
      }
    }
  }

  dispose(): void {
    this.pruneState(new Set())
  }

  private closeMovie(state: MovieState): void {
    state.closed = true
    state.video?.pause()
    state.video?.removeAttribute('src')
  }

  private async movieFrame(nodeId: string, params: Record<string, any>): Promise<CapturedFrame | null> {
    const path = String(params.path ?? '')
    if (!path) return null
    const src = resolveMediaUrl(path)

    let state = this.movies.get(nodeId)
    if (!state || state.src !== src) {
      if (state) this.closeMovie(state)
      const opening: MovieState = { video: null, src, closed: false }
      this.movies.set(nodeId, opening)
      loadVideo(src).then(
        (video) => {
          if (opening.closed) return
          opening.video = video
          this.onReady?.()
        },
        () => {
          // Missing file: the node emits nothing, as before.
        }
      )
      return null
    }
    if (!state.video) return null

    const totalFrames = Math.max(1, Math.floor((state.video.duration || 0) * ASSUMED_FPS))
    const start = Math.max(0, Number(params.start_frame) || 0)
    const rawEnd = Number(params.end_frame) || 0
    const end = rawEnd > 0 ? Math.min(rawEnd, totalFrames - 1) : totalFrames - 1
    const span = Math.max(1, end - start + 1)

    let index = Number(params.scrub_index) || 0
    if (params.loop) index = start + (((index - start) % span) + span) % span
    else index = Math.max(start, Math.min(index, end))

    await seek(state.video, index / ASSUMED_FPS)
    if (!state.video.videoWidth) return null
    const bitmap = await createImageBitmap(state.video)
    return { bitmap, extra: { frame: index, total_frames: totalFrames } }
  }

  private async webcamFrame(nodeId: string): Promise<CapturedFrame | null> {
    const state = this.webcams.get(nodeId)
    if (!state) {
      const opening: WebcamState = { video: null, stream: null, closed: false }
      this.webcams.set(nodeId, opening)
      navigator.mediaDevices.getUserMedia({ video: true }).then(
        async (stream) => {
          if (opening.closed) {
            for (const track of stream.getTracks()) track.stop()
            return
          }
          const video = document.createElement('video')
          video.srcObject = stream
          video.muted = true
          video.playsInline = true
          opening.stream = stream
          await video.play().catch(() => {})
          opening.video = video
          this.onReady?.()
        },
        () => {
          // Denied or no camera: the node emits nothing, as before.
        }
      )
      return null
    }
    if (!state.video?.videoWidth) return null
    const bitmap = await createImageBitmap(state.video)
    return { bitmap, extra: { fps: 30 } }
  }

  /**
   * Captures one frame per movie/webcam node currently in the graph. Nodes
   * whose frame isn't ready yet (video still loading, webcam permission still
   * pending) are simply absent from the result — their worker-side node
   * implementation treats a missing frame as "no output yet".
   */
  async captureFrames(nodes: { id: string; type: string; data?: { params?: Record<string, any> } }[]): Promise<Record<string, CapturedFrame>> {
    this.pruneState(new Set(nodes.filter((n) => n.type === 'input_movie' || n.type === 'input_webcam').map((n) => n.id)))

    const frames: Record<string, CapturedFrame> = {}
    await Promise.all(
      nodes.map(async (node) => {
        try {
          if (node.type === 'input_movie') {
            const frame = await this.movieFrame(node.id, node.data?.params ?? {})
            if (frame) frames[node.id] = frame
          } else if (node.type === 'input_webcam') {
            const frame = await this.webcamFrame(node.id)
            if (frame) frames[node.id] = frame
          }
        } catch {
          // Missing file / denied camera permission — the node just emits nothing this run.
        }
      })
    )
    return frames
  }
}
