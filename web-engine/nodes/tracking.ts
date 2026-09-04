import type { NodeImpl, RunContext } from '../types'
import { drawArrowedLine, drawPolyline, toBgr, toGray } from '../cvUtils'
import { Box, RawTrack, SortTracker } from '../sort'

/**
 * `cv.TERM_CRITERIA_*` are not exported by this OpenCV build (they read back as
 * undefined, which silently becomes 0 in a bitwise OR). COUNT|EPS is 3.
 */
const TERM_COUNT_AND_EPS = 3

/* ------------------------------------------------- background subtraction */

interface Mog2State {
  signature: string
  subtractor: any
}

function mog2For(ctx: RunContext, history: number, threshold: number, shadows: boolean): any {
  const signature = `${history}:${threshold}:${shadows}`
  const state = ctx.state.get(ctx.nodeId) as Mog2State | undefined
  if (state && state.signature === signature) return state.subtractor
  state?.subtractor?.delete?.()
  const subtractor = new ctx.cv.BackgroundSubtractorMOG2(history, threshold, shadows)
  ctx.state.set(ctx.nodeId, { signature, subtractor })
  return subtractor
}

export const bgSubMog2: NodeImpl = (inputs, params, ctx) => {
  const img = (inputs.image ?? inputs.main) as any
  if (!img) return { main: null, mask: null }

  const subtractor = mog2For(
    ctx,
    Math.max(1, Math.round(Number(params.history) || 500)),
    Number(params.threshold) ?? 16,
    params.detect_shadows !== false
  )
  const mask = ctx.track(new ctx.cv.Mat())
  subtractor.apply(img, mask)
  ctx.emit('foreground_px', ctx.cv.countNonZero(mask))
  return { main: mask, mask }
}

/**
 * Sample-consensus background model, standing in for OpenCV's KNN subtractor,
 * which this build does not ship.
 *
 * Same family as the original (Zivkovic & van der Heijden): every pixel keeps a
 * handful of past samples, and is background when at least `k` of them fall
 * within `dist2Threshold`. Updates are random so a sample survives, on average,
 * far longer than the sample count — which is what lets the model tolerate a
 * briefly stopped object without swallowing it.
 */
interface KnnState {
  signature: string
  samples: Uint8Array[]
  width: number
  height: number
  seen: number
  cursor: number
  random: () => number
}

const KNN_SAMPLES = 7
const KNN_MIN_MATCHES = 2

export const bgSubKnn: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const img = (inputs.image ?? inputs.main) as any
  if (!img) return { main: null, mask: null }

  const history = Math.max(1, Math.round(Number(params.history) || 500))
  const dist2 = Number(params.threshold) ?? 400
  const shadows = params.detect_shadows !== false

  const gray = toGray(cv, img)
  const w = gray.cols
  const h = gray.rows
  const n = w * h
  const signature = `${history}:${w}:${h}`

  let state = ctx.state.get(ctx.nodeId) as KnnState | undefined
  if (!state || state.signature !== signature) {
    let seed = 0x9e3779b9
    state = {
      signature,
      samples: Array.from({ length: KNN_SAMPLES }, () => new Uint8Array(n)),
      width: w,
      height: h,
      seen: 0,
      cursor: 0,
      // Deterministic PRNG: the same clip must classify the same way twice.
      random: () => {
        seed = (seed + 0x6d2b79f5) >>> 0
        let t = seed
        t = Math.imul(t ^ (t >>> 15), t | 1)
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296
      },
    }
    ctx.state.set(ctx.nodeId, state)
  }

  const pixels = gray.data
  const mask = ctx.track(new cv.Mat(h, w, cv.CV_8U))
  const out = mask.data

  if (state.seen === 0) {
    // Seed every sample from the first frame, so nothing is foreground yet.
    for (const sample of state.samples) sample.set(pixels)
    out.fill(0)
  } else {
    const radius = Math.sqrt(Math.max(0, dist2))
    for (let i = 0; i < n; i++) {
      const value = pixels[i]
      let close = 0
      for (let s = 0; s < KNN_SAMPLES && close < KNN_MIN_MATCHES; s++) {
        if (Math.abs(value - state.samples[s][i]) <= radius) close++
      }
      if (close >= KNN_MIN_MATCHES) {
        out[i] = 0
      } else if (shadows && value < state.samples[0][i] && value > state.samples[0][i] * 0.5) {
        // Darker than the model but not black: the shadow label MOG2 also uses.
        out[i] = 127
      } else {
        out[i] = 255
      }
    }
  }

  // Conservative update: only background pixels feed the model, at a rate set
  // by `history`, and each write lands in one randomly chosen sample slot.
  const updateChance = 1 / Math.max(1, history / KNN_SAMPLES)
  for (let i = 0; i < n; i++) {
    if (state.seen > 0 && out[i] === 255) continue
    if (state.random() > updateChance) continue
    state.samples[Math.floor(state.random() * KNN_SAMPLES)][i] = pixels[i]
  }
  state.seen++
  state.cursor = (state.cursor + 1) % KNN_SAMPLES

  gray.delete()
  ctx.emit('foreground_px', cv.countNonZero(mask))
  return { main: mask, mask }
}

export const filterBgSubtraction: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const img = (inputs.image ?? inputs.main) as any
  if (!img) return { main: null, mask: null }

  const subtractor = mog2For(
    ctx,
    Math.max(1, Math.round(Number(params.history) || 500)),
    Number(params.threshold) ?? 16,
    params.detectShadows !== false
  )
  const raw = new cv.Mat()
  subtractor.apply(img, raw)

  // Opening clears the speckle a per-pixel model always leaves behind.
  const mask = ctx.track(new cv.Mat())
  const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(3, 3))
  cv.morphologyEx(raw, mask, cv.MORPH_OPEN, kernel)
  kernel.delete()
  raw.delete()

  const colour = toBgr(cv, img)
  const masked = ctx.track(new cv.Mat())
  cv.bitwise_and(colour, colour, masked, mask)
  colour.delete()

  return { main: masked, mask }
}

/* ------------------------------------------------- sparse optical flow (LK) */

const TRACK_HISTORY_LEN = 15

interface LkState {
  prevGray: any | null
  points: [number, number][]
  tracks: [number, number][][]
  colours: [number, number, number][]
  frame: number
  lastReset: number
}

/** Deterministic per-track colour, so a track keeps its colour across frames. */
function trackColour(index: number): [number, number, number] {
  let seed = (index + 1) * 0x9e3779b9
  const next = () => {
    seed = (seed + 0x6d2b79f5) >>> 0
    let t = seed
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) % 256
  }
  return [next(), next(), next()]
}

function detectCorners(cv: any, gray: any, maxCorners: number, quality: number, minDistance: number): [number, number][] {
  const corners = new cv.Mat()
  const noMask = new cv.Mat()
  cv.goodFeaturesToTrack(gray, corners, maxCorners, quality, minDistance, noMask, 7)
  const out: [number, number][] = []
  const data = corners.data32F
  for (let i = 0; i < corners.rows; i++) out.push([data[i * 2], data[i * 2 + 1]])
  corners.delete()
  noMask.delete()
  return out
}

export const opticalFlowLk: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const img = (inputs.image ?? inputs.main) as any
  if (!img) return { main: null, data: { n_tracked: 0, mean_displacement: 0 } }

  const maxCorners = Math.max(10, Math.round(Number(params.max_corners) || 200))
  const quality = Number(params.quality) || 0.3
  const minDistance = Math.max(1, Math.round(Number(params.min_distance) || 7))
  const win = Math.max(1, Math.round(Number(params.win_size) || 15))
  const redetectEvery = Math.max(1, Math.round(Number(params.redetect_every) || 20))
  const drawModes = ['Tracks', 'Arrows', 'Points']
  const rawMode = params.draw
  const drawMode = typeof rawMode === 'string' ? rawMode : drawModes[Math.round(Number(rawMode) || 0)] ?? 'Tracks'

  let state = ctx.state.get(ctx.nodeId) as LkState | undefined
  if (!state) {
    state = { prevGray: null, points: [], tracks: [], colours: [], frame: 0, lastReset: 0 }
    ctx.state.set(ctx.nodeId, state)
  }

  const resetTrigger = Number(params.reset) ? 1 : 0
  if (resetTrigger && !state.lastReset) {
    state.prevGray?.delete?.()
    state.prevGray = null
    state.points = []
    state.tracks = []
    state.colours = []
    state.frame = 0
  }
  state.lastReset = resetTrigger

  const gray = toGray(cv, img)
  const overlay = ctx.track(toBgr(cv, img))

  const needDetect = !state.prevGray || state.points.length === 0 || state.frame % redetectEvery === 0

  let meanDisplacement = 0
  let tracked = 0

  const redetect = () => {
    state!.points = detectCorners(cv, gray, maxCorners, quality, minDistance)
    state!.tracks = state!.points.map((p) => [p])
    state!.colours = state!.points.map((_, i) => trackColour(i))
  }

  if (needDetect) {
    redetect()
  } else {
    const prevMat = cv.matFromArray(state.points.length, 1, cv.CV_32FC2, state.points.flat())
    const nextMat = new cv.Mat()
    const status = new cv.Mat()
    const err = new cv.Mat()
    let ok = true
    try {
      cv.calcOpticalFlowPyrLK(
        state.prevGray,
        gray,
        prevMat,
        nextMat,
        status,
        err,
        new cv.Size(win, win),
        2,
        new cv.TermCriteria(TERM_COUNT_AND_EPS, 10, 0.03)
      )
    } catch {
      ok = false
    }

    if (!ok || status.rows === 0) {
      redetect()
    } else {
      const next = nextMat.data32F
      const flags = status.data
      const keptPoints: [number, number][] = []
      const keptTracks: [number, number][][] = []
      const keptColours: [number, number, number][] = []
      const displacements: number[] = []

      for (let i = 0; i < flags.length; i++) {
        if (!flags[i]) continue
        const nx = next[i * 2]
        const ny = next[i * 2 + 1]
        const [ox, oy] = state.points[i]
        displacements.push(Math.hypot(nx - ox, ny - oy))

        const history = state.tracks[i] ?? []
        history.push([nx, ny])
        if (history.length > TRACK_HISTORY_LEN) history.shift()

        keptPoints.push([nx, ny])
        keptTracks.push(history)
        const colour = state.colours[i] ?? [0, 255, 0]
        keptColours.push(colour)

        const scalar = new cv.Scalar(colour[0], colour[1], colour[2], 255)
        if (drawMode === 'Arrows') {
          drawArrowedLine(cv, overlay, new cv.Point(Math.round(ox), Math.round(oy)), new cv.Point(Math.round(nx), Math.round(ny)), scalar, 2, 0.4)
        } else if (drawMode === 'Points') {
          cv.circle(overlay, new cv.Point(Math.round(nx), Math.round(ny)), 3, scalar, -1)
        }
      }

      if (keptPoints.length === 0) {
        redetect()
      } else {
        state.points = keptPoints
        state.tracks = keptTracks
        state.colours = keptColours
        tracked = keptPoints.length
        if (displacements.length) meanDisplacement = displacements.reduce((a, b) => a + b, 0) / displacements.length

        if (drawMode === 'Tracks') {
          state.tracks.forEach((history, i) => {
            const colour = state!.colours[i]
            const scalar = new cv.Scalar(colour[0], colour[1], colour[2], 255)
            if (history.length >= 2) {
              drawPolyline(cv, overlay, history.map(([x, y]) => ({ x, y })), false, scalar, 2)
            }
            const last = history[history.length - 1]
            cv.circle(overlay, new cv.Point(Math.round(last[0]), Math.round(last[1])), 3, scalar, -1)
          })
        }
      }
    }

    prevMat.delete()
    nextMat.delete()
    status.delete()
    err.delete()
  }

  if (state.points.length) {
    tracked = state.points.length
    if (needDetect && (drawMode === 'Points' || drawMode === 'Tracks')) {
      const green = new cv.Scalar(0, 255, 0, 255)
      for (const [x, y] of state.points) cv.circle(overlay, new cv.Point(Math.round(x), Math.round(y)), 3, green, -1)
    }
  }

  // The previous frame outlives the run, so it is owned here rather than tracked.
  state.prevGray?.delete?.()
  state.prevGray = gray
  state.frame += 1

  const data = { n_tracked: tracked, mean_displacement: Math.round(meanDisplacement * 1000) / 1000 }
  ctx.emit('n_tracked', tracked)
  return { main: overlay, data }
}

/* ------------------------------------------------------------------- SORT */

interface SortState {
  signature: string
  tracker: SortTracker
}

export const trackerSort: NodeImpl = (inputs, params, ctx) => {
  const image = inputs.image as any
  const detections = inputs.detections as any[] | undefined

  const empty: Record<string, unknown> = { main: image ?? null, tracks: [], count: 0 }
  for (let i = 0; i < 5; i++) empty[`track_${i}`] = null

  const maxAge = Math.max(1, Math.round(Number(params.max_age) || 5))
  const minHits = Math.max(1, Math.round(Number(params.min_hits) || 2))
  const iouThreshold = (Number(params.iou_threshold) || 30) / 100
  const signature = `${maxAge}:${minHits}:${iouThreshold}`

  let state = ctx.state.get(ctx.nodeId) as SortState | undefined
  if (!state || state.signature !== signature) {
    state = { signature, tracker: new SortTracker(maxAge, minHits, iouThreshold) }
    ctx.state.set(ctx.nodeId, state)
  }

  if (!image || !Array.isArray(detections) || detections.length === 0) {
    // Still step the tracker: objects disappearing is information too.
    state.tracker.update([], [], [])
    return empty
  }

  const w = image.cols
  const h = image.rows
  const boxes: Box[] = []
  const labels: string[] = []
  const scores: number[] = []
  for (const det of detections) {
    if (!det || typeof det !== 'object') continue
    const xmin = Number(det.xmin) || 0
    const ymin = Number(det.ymin) || 0
    const bw = Number(det.width) || 0
    const bh = Number(det.height) || 0
    boxes.push([xmin * w, ymin * h, (xmin + bw) * w, (ymin + bh) * h])
    labels.push(String(det.label ?? ''))
    scores.push(Number(det.score) || 0)
  }
  if (!boxes.length) return empty

  const raw: RawTrack[] = state.tracker.update(boxes, labels, scores)

  const tracks = raw.map((t) => {
    const x1 = Math.max(0, t.x1 / w)
    const y1 = Math.max(0, t.y1 / h)
    const x2 = Math.min(1, t.x2 / w)
    const y2 = Math.min(1, t.y2 / h)
    return {
      track_id: t.track_id,
      label: `#${t.track_id} ${t.label}`.trim(),
      score: t.score,
      xmin: x1,
      ymin: y1,
      width: x2 - x1,
      height: y2 - y1,
      _type: 'graphics' as const,
      shape: 'rect' as const,
      pts: [
        [x1, y1],
        [x2, y2],
      ],
      thickness: 2,
      // Colour derived from the id, so a track keeps its colour for its lifetime.
      r: Math.round(Math.abs(Math.sin(t.track_id * 0.9)) * 255),
      g: Math.round(Math.abs(Math.sin(t.track_id * 0.9 + 2.1)) * 255),
      b: Math.round(Math.abs(Math.sin(t.track_id * 0.9 + 4.2)) * 255),
    }
  })

  const out: Record<string, unknown> = { main: image, tracks, count: tracks.length }
  for (let i = 0; i < 5; i++) out[`track_${i}`] = tracks[i] ?? null
  return out
}

/* ------------------------------------------------------- track visualizer */

/** Stable, well-separated colour per track id, via a hue rotation. */
function idColour(cv: any, id: number): [number, number, number] {
  const hue = Math.round(id * 47) % 180
  const one = cv.matFromArray(1, 1, cv.CV_8UC3, [hue, 230, 230])
  const bgr = new cv.Mat()
  cv.cvtColor(one, bgr, cv.COLOR_HSV2BGR)
  const out: [number, number, number] = [bgr.data[0], bgr.data[1], bgr.data[2]]
  one.delete()
  bgr.delete()
  return out
}

interface VisualizerState {
  trails: Map<number, [number, number][]>
  activeIds: Set<number>
  pointTrail: [number, number][]
}

const TRAIL_CAP = 120

export const trackerVisualize: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = inputs.image as any
  if (!image) return { main: null }

  const out = ctx.track(toBgr(cv, image))
  const w = out.cols
  const h = out.rows

  let state = ctx.state.get(ctx.nodeId) as VisualizerState | undefined
  if (!state) {
    state = { trails: new Map(), activeIds: new Set(), pointTrail: [] }
    ctx.state.set(ctx.nodeId, state)
  }

  const showTrail = params.show_trail !== false
  const trailLength = Math.max(2, Math.round(Number(params.trail_length) || 30))
  const showId = params.show_id !== false
  const showLabel = params.show_label !== false
  const thickness = Math.max(1, Math.round(Number(params.thickness) || 2))
  const fontScale = (Number(params.font_scale) || 40) / 100
  const fillAlpha = (Number(params.fill_alpha) ?? 10) / 100
  const showPoint = !!params.show_point
  const pointRadius = Math.max(1, Math.round(Number(params.point_radius) || 6))
  const pointUseIdColour = params.point_use_id_color !== false

  const tracks = inputs.tracks as any[] | undefined
  const currentIds = new Set<number>()

  if (Array.isArray(tracks) && tracks.length) {
    for (const track of tracks) {
      if (!track || typeof track !== 'object') continue
      const id = Number(track.track_id) || 0
      const x1 = Math.round((Number(track.xmin) || 0) * w)
      const y1 = Math.round((Number(track.ymin) || 0) * h)
      const x2 = Math.round(((Number(track.xmin) || 0) + (Number(track.width) || 0)) * w)
      const y2 = Math.round(((Number(track.ymin) || 0) + (Number(track.height) || 0)) * h)
      const colour = idColour(cv, id)
      const scalar = new cv.Scalar(colour[0], colour[1], colour[2], 255)
      currentIds.add(id)

      if (fillAlpha > 0) {
        const layer = out.clone()
        cv.rectangle(layer, new cv.Point(x1, y1), new cv.Point(x2, y2), scalar, -1)
        cv.addWeighted(layer, fillAlpha, out, 1 - fillAlpha, 0, out)
        layer.delete()
      }
      cv.rectangle(out, new cv.Point(x1, y1), new cv.Point(x2, y2), scalar, thickness)

      const cx = (x1 + x2) >> 1
      const cy = (y1 + y2) >> 1
      let trail = state.trails.get(id)
      if (!trail) {
        trail = []
        state.trails.set(id, trail)
      }
      trail.push([cx, cy])
      if (trail.length > TRAIL_CAP) trail.shift()

      if (showPoint) {
        const pointColour = pointUseIdColour
          ? scalar
          : new cv.Scalar(Number(params.point_b) || 0, Number(params.point_g) || 0, Number(params.point_r) || 0, 255)
        cv.circle(out, new cv.Point(cx, cy), pointRadius, pointColour, -1)
      }

      if (showTrail && trail.length > 1) {
        const shown = trail.slice(-trailLength)
        for (let k = 1; k < shown.length; k++) {
          // Older segments thin out, so the direction of travel reads at a glance.
          const width = Math.max(1, Math.round((thickness * k) / shown.length))
          cv.line(out, new cv.Point(shown[k - 1][0], shown[k - 1][1]), new cv.Point(shown[k][0], shown[k][1]), scalar, width)
        }
      }

      const parts: string[] = []
      if (showId) parts.push(`#${id}`)
      if (showLabel) {
        const raw = String(track.label ?? '').replace(`#${id}`, '').trim()
        if (raw) parts.push(raw)
        const score = Number(track.score) || 0
        if (score > 0) parts.push(`${Math.round(score * 100)}%`)
      }
      if (parts.length) {
        const text = parts.join(' ')
        // cv.getTextSize is missing from this build; HERSHEY_SIMPLEX advances
        // about 17px per character and stands ~22px tall at scale 1.
        const th = Math.round(22 * fontScale)
        const tw = Math.round(17 * fontScale * text.length)
        cv.rectangle(out, new cv.Point(x1, Math.max(0, y1 - th - 6)), new cv.Point(x1 + tw + 6, y1), scalar, -1)
        cv.putText(out, text, new cv.Point(x1 + 3, Math.max(th, y1 - 4)), cv.FONT_HERSHEY_SIMPLEX, fontScale, new cv.Scalar(255, 255, 255, 255), 1, cv.LINE_AA)
      }
    }
  }

  for (const stale of state.activeIds) if (!currentIds.has(stale)) state.trails.delete(stale)
  state.activeIds = currentIds

  const point = inputs.point as any
  if (point && typeof point === 'object' && Array.isArray(point.pts) && point.pts.length) {
    const relative = point.relative !== false
    const px = Math.round(relative ? point.pts[0][0] * w : point.pts[0][0])
    const py = Math.round(relative ? point.pts[0][1] * h : point.pts[0][1])
    state.pointTrail.push([px, py])
    if (state.pointTrail.length > TRAIL_CAP) state.pointTrail.shift()

    const scalar = new cv.Scalar(Number(point.b) || 0, point.g ?? 255, point.r ?? 80, 255)
    if (showTrail && state.pointTrail.length > 1) {
      const shown = state.pointTrail.slice(-trailLength)
      for (let k = 1; k < shown.length; k++) {
        const width = Math.max(1, Math.round((thickness * k) / shown.length))
        cv.line(out, new cv.Point(shown[k - 1][0], shown[k - 1][1]), new cv.Point(shown[k][0], shown[k][1]), scalar, width)
      }
    }
    cv.circle(out, new cv.Point(px, py), Math.max(1, Math.round(Number(point.thickness) || 8)), scalar, -1)
  } else {
    state.pointTrail = []
  }

  return { main: out }
}

/* --------------------------------------------------- landmark measurements */

interface Landmark {
  x: number
  y: number
}

function landmarksOf(data: unknown): Landmark[] {
  const dict = data as { landmarks?: Landmark[] } | null | undefined
  return dict && Array.isArray(dict.landmarks) ? dict.landmarks : []
}

/** Pixel scale when the node is asked for absolute coordinates. */
function absoluteScale(image: any): [number, number] {
  return image ? [image.cols, image.rows] : [640, 480]
}

export const geomTrackLine: NodeImpl = (inputs, params) => {
  const empty = { distance: 0, draw: null }
  const landmarks = landmarksOf(inputs.data)
  if (!landmarks.length) return empty

  const a = Math.round(Number(params.pt_a) || 0)
  const b = Math.round(Number(params.pt_b) || 0)
  if (a < 0 || a >= landmarks.length || b < 0 || b >= landmarks.length) return empty

  const absolute = !!params.absolute
  const [scaleX, scaleY] = absolute ? absoluteScale(inputs.image) : [1, 1]

  const dx = (landmarks[a].x - landmarks[b].x) * scaleX
  const dy = (landmarks[a].y - landmarks[b].y) * scaleY

  return {
    distance: Math.hypot(dx, dy),
    draw: {
      _type: 'graphics',
      shape: 'line',
      pts: [
        [landmarks[a].x * scaleX, landmarks[a].y * scaleY],
        [landmarks[b].x * scaleX, landmarks[b].y * scaleY],
      ],
      relative: !absolute,
      thickness: Math.round(Number(params.thickness) || 4),
      r: Math.round(Number(params.r) || 0),
      g: Math.round(Number(params.g) ?? 255),
      b: Math.round(Number(params.b) || 0),
    },
  }
}

export const geomTrackPolygon: NodeImpl = (inputs, params) => {
  const empty = { area: 0, draw: null }
  const landmarks = landmarksOf(inputs.data)
  if (!landmarks.length) return empty

  const chosen: Landmark[] = []
  for (let i = 1; i <= 10; i++) {
    const id = Math.round(Number(params[`pt_${i}`] ?? -1))
    if (id >= 0 && id < landmarks.length) chosen.push(landmarks[id])
  }
  if (chosen.length < 3) return empty

  const absolute = !!params.absolute
  const [scaleX, scaleY] = absolute ? absoluteScale(inputs.image) : [1, 1]
  const points = chosen.map((p) => [p.x * scaleX, p.y * scaleY] as [number, number])

  // Shoelace formula over the polygon in the order the points were listed.
  let twiceArea = 0
  for (let i = 0; i < points.length; i++) {
    const j = (i + 1) % points.length
    twiceArea += points[i][0] * points[j][1] - points[j][0] * points[i][1]
  }

  return {
    area: Math.abs(twiceArea) / 2,
    draw: {
      _type: 'graphics',
      shape: 'polygon',
      pts: points,
      relative: !absolute,
      fill: params.fill !== false,
      thickness: Math.round(Number(params.thickness) || 2),
      r: Math.round(Number(params.r) || 0),
      g: Math.round(Number(params.g) ?? 255),
      b: Math.round(Number(params.b) || 0),
    },
  }
}
