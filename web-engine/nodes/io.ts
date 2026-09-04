import type { NodeImpl, RunContext } from '../types'
import { downloadFile } from '../../shims/vfs'
import { drawMatToCanvas, makeCanvas } from '../canvasCompat'
import { toBgr, toGray } from '../cvUtils'
import { buildZip, ZipEntry } from '../zip'

/* ------------------------------------------------------------------ signals */

interface ClockState {
  start: number
  frame: number
  lastReset: number
}

export const signalClock: NodeImpl = (_inputs, params, ctx) => {
  let state = ctx.state.get(ctx.nodeId) as ClockState | undefined
  if (!state) {
    state = { start: performance.now(), frame: 0, lastReset: 0 }
    ctx.state.set(ctx.nodeId, state)
  }

  const resetTrigger = Number(params.reset) ? 1 : 0
  if (resetTrigger && !state.lastReset) {
    state.start = performance.now()
    state.frame = 0
  }
  state.lastReset = resetTrigger

  const speed = Math.max(1, Math.round(Number(params.speed) || 100)) / 100
  const t = ((performance.now() - state.start) / 1000) * speed
  state.frame += 1

  return {
    t,
    ms: t * 1000,
    frame: state.frame,
    sin_t: Math.sin(t),
    cos_t: Math.cos(t),
    display_text: `t=${t.toFixed(3)}s  frame=${state.frame}`,
  }
}

const WAVEFORM_NAMES = ['Sine', 'Square', 'Triangle', 'Sawtooth', 'White Noise', 'Random Walk']

interface GeneratorState {
  start: number
  walk: number
  seed: number
  random: () => number
  spare: number | null
}

/** Deterministic PRNG so a fixed seed reproduces the same noise. */
function makeRandom(seed: number): () => number {
  let state = (seed >>> 0) || 0x9e3779b9
  return () => {
    state = (state + 0x6d2b79f5) >>> 0
    let t = state
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export const signalGenerator: NodeImpl = (inputs, params, ctx) => {
  const waveform = Math.round(Number(params.waveform) || 0)
  let frequency = Math.max(0.0001, (Number(params.frequency) ?? 1) + (Number(params.freq_fine) || 0) / 1000)
  let amplitude = (Number(params.amplitude) ?? 100) / 100
  const offset = (Number(params.offset) || 0) / 100
  const phaseOffset = (Number(params.phase) || 0) / 360
  const duty = (Number(params.duty) ?? 50) / 100
  const walkStep = (Number(params.rw_step) ?? 5) / 100
  const seed = Math.round(Number(params.seed) ?? -1)

  // A wired modulation input overrides the parameter.
  if (inputs.freq_mod !== null && inputs.freq_mod !== undefined) {
    const v = Number(inputs.freq_mod)
    if (Number.isFinite(v)) frequency = Math.max(0.0001, v)
  }
  if (inputs.amp_mod !== null && inputs.amp_mod !== undefined) {
    const v = Number(inputs.amp_mod)
    if (Number.isFinite(v)) amplitude = v
  }

  let state = ctx.state.get(ctx.nodeId) as GeneratorState | undefined
  if (!state) {
    state = { start: performance.now(), walk: 0, seed: seed - 1, random: makeRandom(0), spare: null }
    ctx.state.set(ctx.nodeId, state)
  }
  if (seed !== state.seed) {
    state.seed = seed
    // A negative seed means "unseeded", so vary it per run.
    state.random = makeRandom(seed >= 0 ? seed : (Math.random() * 2 ** 32) >>> 0)
    state.spare = null
  }

  const t = (performance.now() - state.start) / 1000
  // Normalised phase in [0, 1): every waveform below is defined on that.
  const p = (frequency * t + phaseOffset) % 1

  /** Box-Muller, one cached spare per pair — the random walk wants a normal. */
  const normal = () => {
    if (state!.spare !== null) {
      const v = state!.spare
      state!.spare = null
      return v
    }
    const u = Math.max(1e-12, state!.random())
    const v = state!.random()
    const radius = Math.sqrt(-2 * Math.log(u))
    state!.spare = radius * Math.sin(2 * Math.PI * v)
    return radius * Math.cos(2 * Math.PI * v)
  }

  let v: number
  if (waveform === 0) v = Math.sin(2 * Math.PI * p)
  else if (waveform === 1) v = p < duty ? 1 : -1
  else if (waveform === 2) v = 1 - 4 * Math.abs(p - 0.5)
  else if (waveform === 3) v = 2 * p - 1
  else if (waveform === 4) v = state.random() * 2 - 1
  else {
    state.walk += normal() * walkStep
    state.walk = Math.max(-1, Math.min(1, state.walk))
    v = state.walk
  }

  const value = amplitude * v + offset
  const name = WAVEFORM_NAMES[waveform] ?? '?'
  return { value, t, display_text: `${name}  ${frequency.toFixed(3)} Hz → ${value.toFixed(4)}` }
}

export const colorInput: NodeImpl = (_inputs, params) => ({ result: String(params.value ?? '#FF0000') })

/* ---------------------------------------------------------- variable store */

interface VariableState {
  items: (string | number | boolean)[]
  lastAppend: string | null
  lastReplace: string | null
  prevReset: number
  lastClear: number
}

/** Anything non-scalar travels as JSON, so it can be joined into text later. */
function coerce(value: unknown): string | number | boolean {
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') return value
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

export const variableStore: NodeImpl = (inputs, params, ctx) => {
  let state = ctx.state.get(ctx.nodeId) as VariableState | undefined
  if (!state) {
    state = { items: [], lastAppend: null, lastReplace: null, prevReset: 0, lastClear: 0 }
    ctx.state.set(ctx.nodeId, state)
  }

  const emit = () => {
    const separator = String(params.separator ?? '\n')
    const texts = state!.items.map((x) => (typeof x === 'string' ? x : JSON.stringify(x)))
    return {
      text: texts.join(separator),
      list: [...state!.items],
      count: state!.items.length,
      last: texts.length ? texts[texts.length - 1] : '',
    }
  }

  const clearTrigger = Number(params.clear) ? 1 : 0
  let resetNow = clearTrigger === 1 && state.lastClear === 0
  state.lastClear = clearTrigger

  // The reset port fires on a rising edge, so a value held high does not wipe
  // the store on every frame.
  const resetValue = Number(inputs.reset) || 0
  if (resetValue > 0.5 && state.prevReset <= 0.5) resetNow = true
  state.prevReset = resetValue

  if (resetNow) {
    state.items = []
    state.lastAppend = null
    state.lastReplace = null
    return emit()
  }

  const maxItems = Math.max(0, Math.round(Number(params.max_items) || 0))

  const replace = inputs.replace
  if (replace !== null && replace !== undefined) {
    const key = String(coerce(replace))
    if (key !== state.lastReplace) {
      state.lastReplace = key
      state.items = [coerce(replace)]
    }
  }

  const append = inputs.append
  if (append !== null && append !== undefined) {
    const key = String(coerce(append))
    // Only a *changed* value is appended: the graph re-runs every frame, and
    // a steady input would otherwise fill the store instantly.
    if (key !== state.lastAppend) {
      state.lastAppend = key
      state.items.push(coerce(append))
      if (maxItems > 0 && state.items.length > maxItems) {
        state.items = state.items.slice(-maxItems)
      }
    }
  }

  return emit()
}

/* ---------------------------------------------------------------- encoding */

const MIME = { png: 'image/png', jpg: 'image/jpeg' } as const
type ImageFormat = keyof typeof MIME

/** Encodes a Mat via an OffscreenCanvas — cv.imencode is absent from this build. */
async function encodeImage(cv: any, mat: any, format: ImageFormat, quality = 0.95): Promise<Uint8Array> {
  const canvas = makeCanvas(mat.cols, mat.rows)
  drawMatToCanvas(cv, canvas, mat)
  const blob = await canvas.convertToBlob({ type: MIME[format], quality })
  return new Uint8Array(await blob.arrayBuffer())
}

/* ------------------------------------------------------------- save frame */

const SAVE_FORMATS: ImageFormat[] = ['png', 'jpg', 'png', 'png']
const SAVE_EXTENSIONS = ['png', 'jpg', 'tiff', 'bmp']

interface SaveState {
  prevTrigger: number
  lastPath: string
}

export const outputSaveFrame: NodeImpl = async (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = (inputs.image ?? inputs.main) as any

  let state = ctx.state.get(ctx.nodeId) as SaveState | undefined
  if (!state) {
    state = { prevTrigger: 0, lastPath: '' }
    ctx.state.set(ctx.nodeId, state)
  }

  const trigger = Number(inputs.trigger) || 0
  const rising = trigger > 0.5 && state.prevTrigger <= 0.5
  state.prevTrigger = trigger
  const record = !!params.record

  if ((!rising && !record) || !image) return { saved_path: state.lastPath }

  const formatIndex = Math.round(Number(params.format) || 0)
  // TIFF and BMP have no browser encoder, so those choices save a PNG — the
  // lossless option — rather than a file that is not what its name says.
  const format = SAVE_FORMATS[formatIndex] ?? 'png'
  const extension = format === 'jpg' ? 'jpg' : 'png'
  const requested = SAVE_EXTENSIONS[formatIndex] ?? 'png'

  const base = String(params.filename ?? 'frame') || 'frame'
  const stamp = params.auto_timestamp !== false ? `_${Date.now()}` : ''
  const name = `${base}${stamp}.${extension}`

  const quality = Math.max(1, Math.min(100, Math.round(Number(params.quality) || 95))) / 100
  const bytes = await encodeImage(cv, toBgr(cv, image), format, quality)
  downloadFile(name, bytes.buffer as ArrayBuffer, MIME[format])

  state.lastPath = name
  ctx.emit('saved_path', name)
  if (requested !== extension) ctx.emit('format_note', `${requested.toUpperCase()} is not encodable in a browser; saved ${extension.toUpperCase()}`)
  return { saved_path: name }
}

/* ---------------------------------------------------------------- snapshot */

interface SnapshotState {
  lastCapture: number
  lastSave: number
}

export const utilSnapshot: NodeImpl = async (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = (inputs.image ?? inputs.raw_frame ?? inputs.main) as any

  let state = ctx.state.get(ctx.nodeId) as SnapshotState | undefined
  if (!state) {
    state = { lastCapture: 0, lastSave: 0 }
    ctx.state.set(ctx.nodeId, state)
  }

  const capture = Number(params.capture) ? 1 : 0
  const save = Number(params.save_to_disk) ? 1 : 0
  // Both triggers do the same thing here. On the desktop, Capture adds an Image
  // node pointing at the written file; a browser page cannot put a file
  // somewhere a later node could read it back from, so both download the frame.
  const fired = (capture === 1 && state.lastCapture === 0) || (save === 1 && state.lastSave === 0)
  state.lastCapture = capture
  state.lastSave = save

  if (fired && image) {
    const bytes = await encodeImage(cv, toBgr(cv, image), 'png')
    downloadFile(`snap_${Date.now()}.png`, bytes.buffer as ArrayBuffer, MIME.png)
  }

  return { main: image ?? null }
}

/* ------------------------------------------------------------ export crops */

interface CropItem {
  id: number
  box: [number, number, number, number]
  contour: [number, number][] | null
}

/** Normalised or pixel box → pixel corners. */
function boxToCorners(box: Record<string, any>, w: number, h: number): [number, number, number, number] {
  const x = Number(box.xmin) || 0
  const y = Number(box.ymin) || 0
  const bw = Number(box.width) || 0
  const bh = Number(box.height) || 0
  // Boxes in this app are normalised; anything larger than 1 is already pixels.
  const scaleX = x <= 1 && bw <= 1 ? w : 1
  const scaleY = y <= 1 && bh <= 1 ? h : 1
  return [Math.round(x * scaleX), Math.round(y * scaleY), Math.round((x + bw) * scaleX), Math.round((y + bh) * scaleY)]
}

function contourToPixels(entry: unknown, w: number, h: number): [number, number][] | null {
  const dict = entry as { pts?: unknown[]; relative?: boolean } | null
  if (!dict || !Array.isArray(dict.pts) || dict.pts.length < 3) return null
  const relative = dict.relative !== false
  return dict.pts
    .map((raw) => {
      const p = raw as number[]
      return Array.isArray(p) && p.length >= 2
        ? ([Math.round(relative ? p[0] * w : p[0]), Math.round(relative ? p[1] * h : p[1])] as [number, number])
        : null
    })
    .filter((p): p is [number, number] => p !== null)
}

export const exportCrops: NodeImpl = async (inputs, params, ctx) => {
  const cv = ctx.cv
  const image = (inputs.image ?? inputs.main) as any

  const trigger = Number(params.export_trigger) ? 1 : 0
  const state = (ctx.state.get(ctx.nodeId) as { last: number } | undefined) ?? { last: 0 }
  const rising = trigger === 1 && state.last === 0
  state.last = trigger
  ctx.state.set(ctx.nodeId, state)

  if (!rising || !image) return { count: 0 }

  const w = image.cols
  const h = image.rows
  const source = Math.round(Number(params.source) || 0)
  const boxes = Array.isArray(inputs.boxes_list) ? (inputs.boxes_list as Record<string, any>[]) : []
  const contours = Array.isArray(inputs.contours) ? (inputs.contours as unknown[]) : []

  const useContours = source === 2 || (source === 0 && contours.length > 0)
  const items: CropItem[] = []
  if (useContours) {
    contours.forEach((entry, i) => {
      const pts = contourToPixels(entry, w, h)
      if (!pts) return
      const xs = pts.map((p) => p[0])
      const ys = pts.map((p) => p[1])
      items.push({ id: i, box: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)], contour: pts })
    })
  } else {
    boxes.forEach((box, i) => {
      if (!box || typeof box !== 'object') return
      items.push({ id: Number(box.id) ?? i, box: boxToCorners(box, w, h), contour: null })
    })
  }
  if (!items.length) return { count: 0 }

  const labels = new Map<number, string>()
  if (Array.isArray(inputs.labels_list)) {
    ;(inputs.labels_list as Record<string, any>[]).forEach((entry, i) => {
      if (entry && typeof entry === 'object') labels.set(Number(entry.id) ?? i, String(entry.label ?? entry.class ?? ''))
    })
  }

  const pad = Math.max(0, Math.round(Number(params.pad) ?? 5))
  const format: ImageFormat = Math.round(Number(params.format) || 0) === 1 ? 'jpg' : 'png'
  // JPEG has no alpha, so a masked cut only makes sense as PNG.
  const masked = Math.round(Number(params.cut_mode) || 0) === 1 && format === 'png'
  const naming = Math.round(Number(params.naming) || 0)
  const byClass = naming === 2 || (naming === 0 && labels.size > 0)
  const prefix = String(params.prefix ?? 'crop') || 'crop'

  const source4 = masked ? new cv.Mat() : null
  if (source4) cv.cvtColor(toBgr(cv, image), source4, cv.COLOR_BGR2BGRA)
  const colour = masked ? source4! : toBgr(cv, image)

  const entries: ZipEntry[] = []
  for (const item of items) {
    const xs = Math.max(0, item.box[0] - pad)
    const ys = Math.max(0, item.box[1] - pad)
    const xe = Math.min(w, item.box[2] + pad)
    const ye = Math.min(h, item.box[3] + pad)
    if (xe <= xs || ye <= ys) continue

    const rect = new cv.Rect(xs, ys, xe - xs, ye - ys)
    const crop = colour.roi(rect).clone()

    if (masked && item.contour) {
      // Everything outside the contour becomes fully transparent.
      const stencil = cv.Mat.zeros(crop.rows, crop.cols, cv.CV_8U)
      const shifted = item.contour.flatMap(([px, py]) => [px - xs, py - ys])
      const poly = cv.matFromArray(item.contour.length, 1, cv.CV_32SC2, shifted)
      const vector = new cv.MatVector()
      vector.push_back(poly)
      cv.fillPoly(stencil, vector, new cv.Scalar(255))
      vector.delete()
      poly.delete()
      const bits = stencil.data
      const data = crop.data
      for (let p = 0; p < bits.length; p++) if (!bits[p]) data[p * 4 + 3] = 0
      stencil.delete()
    }

    const label = labels.get(item.id) ?? ''
    const folder = byClass && label ? `${label.replace(/[^\w.-]+/g, '_')}/` : ''
    entries.push({ name: `${folder}${prefix}_${item.id}.${format}`, bytes: await encodeImage(cv, crop, format) })
    crop.delete()
  }

  if (masked) source4!.delete()
  else colour.delete()

  if (entries.length) {
    const zip = buildZip(entries)
    downloadFile(`${prefix}_${Date.now()}.zip`, zip.buffer as ArrayBuffer, 'application/zip')
  }

  ctx.emit('count', entries.length)
  return { count: entries.length }
}

/* --------------------------------------------------------- object extractor */

const OBJ_PALETTE: [number, number, number][] = [
  [220, 60, 60], [60, 180, 75], [67, 133, 255], [255, 165, 0],
  [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60],
  [250, 190, 212], [0, 128, 128], [220, 190, 255], [170, 110, 40],
  [128, 0, 0], [128, 128, 0], [0, 0, 128], [128, 128, 128],
]

export const objExtractor: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const objects = inputs.objects as Record<string, any>[] | undefined
  const maskIn = inputs.mask as any
  const image = inputs.image as any
  const minArea = Math.max(0, Math.round(Number(params.min_area) || 0))

  const reference = maskIn ?? image
  const w = reference ? reference.cols : 512
  const h = reference ? reference.rows : 512

  const labelsMap = ctx.track(cv.Mat.zeros(h, w, cv.CV_32S))
  const stats: Record<string, number>[] = []
  let labelId = 0

  const hasObjects = Array.isArray(objects) && objects.length > 0
  if (hasObjects) {
    for (const obj of objects!) {
      if (!obj || typeof obj !== 'object') continue
      const relative = obj.relative === true
      const pts = Array.isArray(obj.pts) ? obj.pts : []

      if (obj.shape === 'circle') {
        if (!pts.length) continue
        const cx = Math.trunc(relative ? pts[0][0] * w : pts[0][0])
        const cy = Math.trunc(relative ? pts[0][1] * h : pts[0][1])
        const r = Math.trunc(Number(obj.radius) || 0)
        if (r <= 0) continue
        const area = Math.PI * r * r
        if (area < minArea) continue
        labelId += 1
        cv.circle(labelsMap, new cv.Point(cx, cy), r, new cv.Scalar(labelId), -1)
        stats.push({ id: labelId, area: Math.round(area * 10) / 10, cx, cy, diameter: Math.round(2 * r * 100) / 100 })
      } else if (pts.length >= 3) {
        const flat = pts.flatMap((p: number[]) => [
          Math.trunc(relative ? p[0] * w : p[0]),
          Math.trunc(relative ? p[1] * h : p[1]),
        ])
        const poly = cv.matFromArray(pts.length, 1, cv.CV_32SC2, flat)
        const area = cv.contourArea(poly)
        if (area < minArea) {
          poly.delete()
          continue
        }
        labelId += 1
        const vector = new cv.MatVector()
        vector.push_back(poly)
        cv.fillPoly(labelsMap, vector, new cv.Scalar(labelId))
        const m = cv.moments(poly)
        const cx = m.m00 > 0 ? Math.trunc(m.m10 / m.m00) : 0
        const cy = m.m00 > 0 ? Math.trunc(m.m01 / m.m00) : 0
        vector.delete()
        poly.delete()
        stats.push({
          id: labelId,
          area: Math.round(area * 10) / 10,
          cx,
          cy,
          diameter: Math.round(2 * Math.sqrt(area / Math.PI) * 100) / 100,
        })
      }
    }
  }

  const labelData = labelsMap.data32S

  if (maskIn) {
    let binary = toGray(cv, maskIn)
    if (binary.cols !== w || binary.rows !== h) {
      const resized = new cv.Mat()
      cv.resize(binary, resized, new cv.Size(w, h), 0, 0, cv.INTER_NEAREST)
      binary.delete()
      binary = resized
    }
    const thresholded = new cv.Mat()
    cv.threshold(binary, thresholded, 0, 255, cv.THRESH_BINARY)
    binary.delete()

    const ccLabels = new cv.Mat()
    const ccStats = new cv.Mat()
    const centroids = new cv.Mat()
    const count = cv.connectedComponentsWithStats(thresholded, ccLabels, ccStats, centroids)
    const cc = ccLabels.data32S

    for (let i = 1; i < count; i++) {
      const area = ccStats.intAt(i, cv.CC_STAT_AREA)
      if (area < Math.max(minArea, 1)) continue

      let covered = 0
      let sumX = 0
      let sumY = 0
      for (let p = 0; p < cc.length; p++) {
        if (cc[p] !== i) continue
        if (labelData[p] !== 0) covered++
        sumX += p % w
        sumY += (p / w) | 0
      }
      // A region already claimed by an object is not a second object.
      if (hasObjects && covered > area * 0.5) continue

      labelId += 1
      for (let p = 0; p < cc.length; p++) if (cc[p] === i) labelData[p] = labelId
      stats.push({
        id: labelId,
        area,
        cx: Math.trunc(sumX / area),
        cy: Math.trunc(sumY / area),
        diameter: Math.round(2 * Math.sqrt(area / Math.PI) * 100) / 100,
      })
    }

    ccLabels.delete()
    ccStats.delete()
    centroids.delete()
    thresholded.delete()
  }

  const preview = ctx.track(new cv.Mat(h, w, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 255)))
  const view = preview.data
  for (let p = 0; p < labelData.length; p++) {
    const id = labelData[p]
    if (!id) continue
    const [r, g, b] = OBJ_PALETTE[(id - 1) % OBJ_PALETTE.length]
    view[p * 3] = r
    view[p * 3 + 1] = g
    view[p * 3 + 2] = b
  }

  const areas = stats.map((s) => s.area)
  const diameters = stats.map((s) => s.diameter)
  const meanArea = areas.length ? areas.reduce((a, b) => a + b, 0) / areas.length : 0
  const meanDiameter = diameters.length ? diameters.reduce((a, b) => a + b, 0) / diameters.length : 0
  let stdDiameter = 0
  if (diameters.length > 1) {
    stdDiameter = Math.sqrt(diameters.reduce((a, b) => a + (b - meanDiameter) ** 2, 0) / diameters.length)
  }
  // Coefficient of variation of the diameters — the anisocytosis figure.
  const cvDiameter = meanDiameter > 0 ? (stdDiameter / meanDiameter) * 100 : 0

  return {
    count: stats.length,
    labels: labelsMap,
    main: preview,
    stats: {
      count: stats.length,
      objects: stats,
      mean_area: Math.round(meanArea * 10) / 10,
      mean_diam: Math.round(meanDiameter * 100) / 100,
      cv_diam: Math.round(cvDiameter * 10) / 10,
    },
  }
}
