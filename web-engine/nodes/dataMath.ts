import type { NodeImpl } from '../types'

export const dataCoordCombine: NodeImpl = (inputs) => {
  const x = inputs.x
  const y = inputs.y
  if (typeof x !== 'number' || typeof y !== 'number') return { dict_out: null }
  return {
    dict_out: {
      x,
      y,
      xmin: x,
      ymin: y,
      width: typeof inputs.w === 'number' ? inputs.w : 0,
      height: typeof inputs.h === 'number' ? inputs.h : 0,
    },
  }
}

export const dataCoordSplitter: NodeImpl = (inputs) => {
  const value = inputs.dict_in as Record<string, number> | null | undefined
  if (!value) return { a: 0, b: 0 }
  return { a: value.x ?? value.xmin ?? 0, b: value.y ?? value.ymin ?? 0 }
}

export const dataListSelector: NodeImpl = (inputs, params) => {
  const list = inputs.list_in
  const index = Number(params.index) || 0
  if (!Array.isArray(list) || index < 0 || index >= list.length) return { item_out: null }
  return { item_out: list[index] }
}

export const pluginDictGet: NodeImpl = (inputs, params, ctx) => {
  const source = inputs.dict_in
  if (!source || typeof source !== 'object' || Array.isArray(source)) {
    return { val_1: null, val_2: null, val_3: null, scalar_1: 0, dict_1: null }
  }
  const dict = source as Record<string, unknown>
  const read = (key: unknown) => {
    const name = String(key ?? '').trim()
    return name ? dict[name] : undefined
  }

  const v1 = read(params.key_1)
  const v2 = read(params.key_2)
  const v3 = read(params.key_3)

  const numeric = typeof v1 === 'number' ? v1 : Number(v1)
  const scalar = Number.isFinite(numeric) ? numeric : 0
  ctx.emit('display_value', String(v1 ?? '—'))

  return {
    val_1: v1 ?? null,
    val_2: v2 ?? null,
    val_3: v3 ?? null,
    scalar_1: scalar,
    dict_1: v1 && typeof v1 === 'object' && !Array.isArray(v1) ? v1 : null,
  }
}

/** Disconnected scalar inputs fall back to the node's own param, as on desktop. */
function operand(inputs: Record<string, unknown>, key: string, params: Record<string, any>, paramKey: string): number {
  const wired = inputs[key]
  if (typeof wired === 'number') return wired
  return Number(params[paramKey]) || 0
}

function binary(fn: (a: number, b: number) => number): NodeImpl {
  return (inputs, params, ctx) => {
    const result = fn(operand(inputs, 'a', params, 'value_a'), operand(inputs, 'b', params, 'value_b'))
    ctx.emit('display_value', formatNumber(result))
    return { result }
  }
}

function unary(fn: (a: number) => number): NodeImpl {
  return (inputs, params, ctx) => {
    const result = fn(operand(inputs, 'a', params, 'value_a'))
    ctx.emit('display_value', formatNumber(result))
    return { result }
  }
}

function formatNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(3)
}

export const mathAdd = binary((a, b) => a + b)
export const mathSub = binary((a, b) => a - b)
export const mathMul = binary((a, b) => a * b)
export const mathDiv = binary((a, b) => (b !== 0 ? a / b : 0))
export const mathMod = binary((a, b) => (b !== 0 ? a % b : 0))
export const mathMin = binary(Math.min)
export const mathMax = binary(Math.max)
export const mathPow = binary((a, b) => a ** b)
export const mathAbs = unary(Math.abs)
export const mathRound = unary(Math.round)
export const mathSin = unary(Math.sin)
export const mathCos = unary(Math.cos)

export const mathClamp: NodeImpl = (inputs, params, ctx) => {
  const value = typeof inputs.val === 'number' ? inputs.val : Number(params.val) || 0
  const low = typeof inputs.min === 'number' ? inputs.min : Number(params.min) || 0
  const high = typeof inputs.max === 'number' ? inputs.max : Number(params.max) || 1
  const result = Math.max(low, Math.min(high, value))
  ctx.emit('display_value', formatNumber(result))
  return { result }
}

export const mathDistance: NodeImpl = (inputs, _params, ctx) => {
  const p1 = (inputs.p1 as Record<string, number>) ?? {}
  const p2 = (inputs.p2 as Record<string, number>) ?? {}
  const x1 = p1.x ?? p1.xmin ?? 0
  const y1 = p1.y ?? p1.ymin ?? 0
  const x2 = p2.x ?? p2.xmin ?? 0
  const y2 = p2.y ?? p2.ymin ?? 0
  const result = Math.hypot(x2 - x1, y2 - y1)
  ctx.emit('display_value', formatNumber(result))
  return { result }
}

export const scalarInput: NodeImpl = (_inputs, params) => {
  const raw = Number(params.value) || 0
  return { value: Number(params.format) === 0 ? Math.round(raw) : raw }
}

// --- list / dict utilities ---------------------------------------------------

interface RegionItem {
  pts?: [number, number][]
  xmin?: number
  ymin?: number
  width?: number
  height?: number
  confidence?: number
  score?: number
  [key: string]: unknown
}

function canonicalCorners(ptsRaw: [number, number][]): [number, number][] {
  if (!ptsRaw || ptsRaw.length < 4) return ptsRaw
  const pts = ptsRaw.slice(0, 4)
  const sums = pts.map(([x, y]) => x + y)
  const diffs = pts.map(([x, y]) => y - x)
  const tl = pts[sums.indexOf(Math.min(...sums))]
  const br = pts[sums.indexOf(Math.max(...sums))]
  const tr = pts[diffs.indexOf(Math.min(...diffs))]
  const bl = pts[diffs.indexOf(Math.max(...diffs))]
  return [tl, tr, br, bl]
}

export const listRegionSelect: NodeImpl = (inputs, params) => {
  const raw = inputs.list_in
  if (!Array.isArray(raw)) return { item: null, pts: [], list_out: [], count: 0 }

  const requirePts = params.require_pts !== false
  const minArea = Number(params.min_area) || 0

  const items = (raw as RegionItem[]).filter((it) => {
    if (!it || typeof it !== 'object') return false
    const area = (it.width ?? 0) * (it.height ?? 0)
    if (area < minArea) return false
    if (requirePts && (it.pts?.length ?? 0) !== 4) return false
    return true
  })

  if (items.length === 0) return { item: null, pts: [], list_out: [], count: 0 }

  const sortBy = Number(params.sort_by) ?? 1
  if (sortBy === 1) items.sort((a, b) => (b.width ?? 0) * (b.height ?? 0) - (a.width ?? 0) * (a.height ?? 0))
  else if (sortBy === 2) items.sort((a, b) => (a.width ?? 0) * (a.height ?? 0) - (b.width ?? 0) * (b.height ?? 0))
  else if (sortBy === 3) items.sort((a, b) => (b.confidence ?? b.score ?? 0) - (a.confidence ?? a.score ?? 0))

  const idx = Math.max(0, Math.min(Math.round(Number(params.index) || 0), items.length - 1))
  const selected = items[idx]

  let ptsOut: [number, number][]
  if ((selected.pts?.length ?? 0) === 4) {
    ptsOut = canonicalCorners(selected.pts as [number, number][])
  } else {
    const x0 = selected.xmin ?? 0
    const y0 = selected.ymin ?? 0
    const x1 = x0 + (selected.width ?? 0)
    const y1 = y0 + (selected.height ?? 0)
    ptsOut = canonicalCorners([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
  }

  return { item: selected, pts: ptsOut, list_out: items, count: items.length }
}

const LIST_OPS = ['sort', 'sort desc', 'reverse', 'unique', 'flatten', 'slice', 'append', 'length', 'contains']

export const dataListOps: NodeImpl = (inputs, params) => {
  let list: unknown[] = Array.isArray(inputs.list_in) ? (inputs.list_in as unknown[]) : []
  const value = inputs.value
  const opIdx = Number(params.operation) || 0
  const op = LIST_OPS[opIdx] ?? 'sort'

  let listOut = list
  let valueOut: unknown = null

  const sortKey = (v: unknown) => (v === null || v === undefined ? [1, 0] : [0, v])
  const compare = (a: unknown, b: unknown) => {
    const [an, av] = sortKey(a)
    const [bn, bv] = sortKey(b)
    if (an !== bn) return (an as number) - (bn as number)
    if (typeof av === 'number' && typeof bv === 'number') return av - bv
    return String(av).localeCompare(String(bv))
  }

  try {
    if (op === 'sort') listOut = [...list].sort(compare)
    else if (op === 'sort desc') listOut = [...list].sort((a, b) => compare(b, a))
    else if (op === 'reverse') listOut = [...list].reverse()
    else if (op === 'unique') {
      const seen = new Set<string>()
      listOut = list.filter((item) => {
        const key = JSON.stringify(item)
        if (seen.has(key)) return false
        seen.add(key)
        return true
      })
    } else if (op === 'flatten') {
      listOut = list.flatMap((item) => (Array.isArray(item) ? item : [item]))
    } else if (op === 'slice') {
      const start = Math.round(Number(params.slice_start) || 0)
      const end = Math.round(Number(params.slice_end) ?? 10)
      const step = Math.round(Number(params.slice_step) || 1) || 1
      if (step === 1) {
        listOut = list.slice(start, end)
      } else {
        listOut = []
        for (let i = start; step > 0 ? i < end : i > end; i += step) {
          if (i < 0 || i >= list.length) continue
          listOut.push(list[i])
        }
      }
    } else if (op === 'append') {
      listOut = value !== null && value !== undefined ? [...list, value] : list
    } else if (op === 'length') {
      valueOut = list.length
      listOut = list
    } else if (op === 'contains') {
      valueOut = list.includes(value) ? 1 : 0
      listOut = list
    }
  } catch {
    // fall through with defaults
  }

  return { list_out: listOut, value_out: valueOut }
}

export const utilLandmarkSelector: NodeImpl = (inputs, params) => {
  const data = inputs.data as Record<string, unknown> | undefined
  if (!data || !Array.isArray((data as any).landmarks)) return { data: {} }

  const lms = (data as any).landmarks as { x: number; y: number }[]
  const indicesStr = String(params.indices ?? '11,12,24,23')
  const idxList = indicesStr
    .split(/[,\s]+/)
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => Number.isFinite(n))

  const filtered = idxList.filter((i) => i >= 0 && i < lms.length).map((i) => lms[i])
  if (filtered.length === 0) return { data: {} }

  const xs = filtered.map((p) => p.x)
  const ys = filtered.map((p) => p.y)
  const xmin = Math.min(...xs)
  const ymin = Math.min(...ys)
  const xmax = Math.max(...xs)
  const ymax = Math.max(...ys)

  return {
    data: {
      ...data,
      landmarks: filtered,
      pts: filtered.map((p) => [p.x, p.y]),
      xmin,
      ymin,
      width: xmax - xmin,
      height: ymax - ymin,
    },
  }
}

export const utilDictMerge: NodeImpl = (inputs) => {
  const result: Record<string, unknown> = {}
  for (const value of Object.values(inputs)) {
    if (value && typeof value === 'object' && !Array.isArray(value)) Object.assign(result, value)
  }
  return { main: result }
}

export const dataGroupDicts: NodeImpl = (inputs) => {
  const grouped: Record<string, unknown>[] = []
  for (let i = 1; i <= 4; i++) {
    const d = inputs[`dict${i}`]
    if (d && typeof d === 'object' && !Array.isArray(d)) grouped.push(d as Record<string, unknown>)
  }
  return { main: grouped.length > 0 ? grouped : null }
}

export const dictBuilder: NodeImpl = (inputs, params) => {
  const out: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(inputs)) {
    if (key === 'raw_frame' || value === null || value === undefined) continue
    const rename = params[`name_${key}`]
    const name = rename !== undefined && rename !== null && String(rename).trim() !== '' ? String(rename).trim() : key
    out[name] = value
  }
  return { dict: out }
}
