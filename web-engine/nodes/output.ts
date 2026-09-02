import type { NodeImpl } from '../types'
import { isMat, toBgr } from '../cvUtils'
import { downloadFile } from '../../shims/vfs'

export const outputDisplay: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const sources = [inputs.main, inputs.mask_in, inputs.flow_in].filter(isMat) as any[]
  if (sources.length === 0) return { main: null }
  if (sources.length === 1) return { main: sources[0] }

  const panels = sources.map((s) => ctx.track(toBgr(cv, s)))
  const mode = Number(params.mode) || 0
  const gap = Math.max(0, Number(params.gap) ?? 2)

  if (mode === 5) {
    const alpha = Number(params.alpha) ?? 0.5
    const blended = ctx.track(new cv.Mat())
    const second = ctx.track(resizeTo(cv, panels[1], panels[0].cols, panels[0].rows))
    cv.addWeighted(panels[0], 1 - alpha, second, alpha, 0, blended)
    return { main: blended }
  }

  if (mode === 4) {
    const split = Math.max(0, Math.min(100, Number(params.split_pos) ?? 50))
    const out = ctx.track(panels[0].clone())
    const second = ctx.track(resizeTo(cv, panels[1], out.cols, out.rows))
    const boundary = Math.round((split / 100) * out.cols)
    if (boundary < out.cols) {
      const width = out.cols - boundary
      const target = out.roi(new cv.Rect(boundary, 0, width, out.rows))
      const source = second.roi(new cv.Rect(boundary, 0, width, out.rows))
      source.copyTo(target)
      target.delete()
      source.delete()
    }
    return { main: out }
  }

  if (mode === 3) {
    const out = ctx.track(panels[0].clone())
    const insetWidth = Math.max(1, Math.round(out.cols / 3))
    const insetHeight = Math.max(1, Math.round((insetWidth * panels[1].rows) / panels[1].cols))
    const inset = ctx.track(resizeTo(cv, panels[1], insetWidth, insetHeight))
    if (insetHeight < out.rows && insetWidth < out.cols) {
      const target = out.roi(new cv.Rect(out.cols - insetWidth - gap, out.rows - insetHeight - gap, insetWidth, insetHeight))
      inset.copyTo(target)
      target.delete()
    }
    return { main: out }
  }

  const vertical = mode === 1
  const columns = mode === 2 ? Math.max(1, Number(params.grid_cols) || 2) : vertical ? 1 : panels.length
  const rows = Math.ceil(panels.length / columns)

  // Every tile is normalised to the first panel's size so the grid stays aligned.
  const cellWidth = panels[0].cols
  const cellHeight = panels[0].rows
  const canvasWidth = columns * cellWidth + (columns - 1) * gap
  const canvasHeight = rows * cellHeight + (rows - 1) * gap
  const out = ctx.track(new cv.Mat(canvasHeight, canvasWidth, cv.CV_8UC3, new cv.Scalar(20, 22, 28)))

  panels.forEach((panel, i) => {
    const scaled = ctx.track(resizeTo(cv, panel, cellWidth, cellHeight))
    const col = i % columns
    const row = Math.floor(i / columns)
    const target = out.roi(new cv.Rect(col * (cellWidth + gap), row * (cellHeight + gap), cellWidth, cellHeight))
    scaled.copyTo(target)
    target.delete()
  })

  return { main: out }
}

function resizeTo(cv: any, mat: any, width: number, height: number): any {
  if (mat.cols === width && mat.rows === height) return mat.clone()
  const out = new cv.Mat()
  cv.resize(mat, out, new cv.Size(width, height), 0, 0, cv.INTER_AREA)
  return out
}

export const dataInspector: NodeImpl = (inputs, _params, ctx) => {
  const value = inputs.data
  ctx.emit('display_value', formatValue(value))
  return { main: inputs.image ?? null, data_out: value ?? null }
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(3)
  if (typeof value === 'string' || typeof value === 'boolean') return String(value)
  if (Array.isArray(value)) return `${value.length} éléments`
  if (isMat(value)) return `image ${(value as any).cols}×${(value as any).rows}`
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

interface CsvState {
  rows: Record<string, unknown>[]
  columns: string[]
  wasRecording: boolean
  lastSnapshot: unknown
}

const SCALAR_TYPES = new Set(['number', 'string', 'boolean'])

export const utilCsvExport: NodeImpl = (inputs, params, ctx) => {
  let state: CsvState = ctx.state.get(ctx.nodeId)
  if (!state) {
    state = { rows: [], columns: [], wasRecording: false, lastSnapshot: null }
    ctx.state.set(ctx.nodeId, state)
  }

  // Dynamic ports: every connected scalar becomes a column, keyed by port name.
  const scalars: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(inputs)) {
    if (value === null || value === undefined) continue
    if (SCALAR_TYPES.has(typeof value)) scalars[key] = value
  }

  // A table on any port exports whole, rather than one row per run.
  const table = Object.values(inputs).find((v) => Array.isArray(v) && v.length > 0 && typeof v[0] === 'object') as
    | Record<string, unknown>[]
    | undefined

  const recording = !!params.record
  if (recording && !state.wasRecording) {
    state.rows = []
    state.columns = Object.keys(scalars)
  }
  state.wasRecording = recording

  if (recording && Object.keys(scalars).length > 0) {
    state.rows.push({ tick: state.rows.length, ...scalars })
  }

  const snapshot = params.snapshot
  const snapshotRequested = !!snapshot && snapshot !== state.lastSnapshot
  state.lastSnapshot = snapshot

  if (snapshotRequested) {
    const rows = table ?? (state.rows.length > 0 ? state.rows : [{ tick: 0, ...scalars }])
    const filename = `${params.filename || 'capture'}${params.auto_timestamp === false ? '' : `_${Date.now()}`}.csv`
    downloadFile(filename, toCsv(rows), 'text/csv')
    ctx.emit('display_value', `${rows.length} lignes exportées`)
  } else {
    ctx.emit('display_value', recording ? `enregistrement — ${state.rows.length} lignes` : `${state.rows.length} lignes en mémoire`)
  }

  return {}
}

function toCsv(rows: Record<string, unknown>[]): string {
  if (rows.length === 0) return ''
  const columns = [...new Set(rows.flatMap((r) => Object.keys(r)))]
  const escape = (v: unknown) => {
    if (v === null || v === undefined) return ''
    const text = typeof v === 'number' ? String(v) : String(v)
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
  }
  return [columns.join(','), ...rows.map((row) => columns.map((c) => escape(row[c])).join(','))].join('\n')
}
