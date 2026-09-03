import type { NodeImpl } from '../types'
import { downloadFile } from '../../shims/vfs'

/**
 * The desktop node runs Python with numpy/OpenCV in a restricted namespace.
 * There is no Python in the browser, so this evaluates JavaScript instead,
 * keeping the same contract: inputs arrive as `a`, `b`, `c` …, and any variable
 * named `out_*` becomes an output port. Scripts written for the desktop node
 * will not run here — the editor shows a banner saying so.
 */
export const logicPython: NodeImpl = (inputs, params, ctx) => {
  const code = String(params.code ?? '')
  const inputNames = Object.keys(inputs).sort()
  const outputNames = [...code.matchAll(/\bout_([a-z0-9_]+)\s*=/gi)].map((m) => `out_${m[1]}`)
  const uniqueOutputs = [...new Set(outputNames)]

  if (uniqueOutputs.length === 0) return { out_a: null }

  const declarations = uniqueOutputs.map((name) => `let ${name} = null;`).join('\n')
  const returns = `return { ${uniqueOutputs.join(', ')} };`

  try {
    const fn = new Function(...inputNames, 'state', `"use strict";\n${declarations}\n${code}\n${returns}`)
    const nodeState = ctx.state.get(`${ctx.nodeId}:script`) ?? {}
    ctx.state.set(`${ctx.nodeId}:script`, nodeState)
    const result = fn(...inputNames.map((n) => inputs[n]), nodeState)
    ctx.emit('error', '')
    return result
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    ctx.emit('error', message)
    return Object.fromEntries(uniqueOutputs.map((name) => [name, null]))
  }
}

export const canvasNote: NodeImpl = (inputs, params) => {
  const incoming = inputs.text
  if (incoming === undefined || incoming === null) return { text_out: String(params.text ?? '') }
  return { text_out: String(incoming) }
}

export const canvasFrame: NodeImpl = () => ({})

// --- logic --------------------------------------------------------------

interface CollectState {
  list: unknown[]
  lastCondition: boolean
  lastReset: number
  lastExport: number
  cooldown: number
}

export const logicCollect: NodeImpl = (inputs, params, ctx) => {
  let state: CollectState = ctx.state.get(ctx.nodeId)
  if (!state) {
    state = { list: [], lastCondition: false, lastReset: 0, lastExport: 0, cooldown: 0 }
    ctx.state.set(ctx.nodeId, state)
  }

  const always = !!params.always
  const condition = always ? true : !!inputs.condition
  const value = inputs.value
  const mode = Number(params.mode) || 0
  const interval = Math.max(1, Math.round(Number(params.interval) || 30))
  const resetTrig = Number(params.reset) || 0
  const exportTrig = Number(params.export) || 0

  if (resetTrig === 1 && state.lastReset === 0) {
    state.list = []
    state.cooldown = 0
  }
  state.lastReset = resetTrig

  let shouldAppend: boolean
  if (mode === 0) {
    shouldAppend = condition
  } else if (mode === 1) {
    shouldAppend = condition && !state.lastCondition
  } else {
    if (state.cooldown > 0) {
      state.cooldown -= 1
      shouldAppend = false
    } else if (condition) {
      shouldAppend = true
      state.cooldown = interval
    } else {
      shouldAppend = false
    }
  }

  if (shouldAppend && value !== null && value !== undefined) state.list.push(value)
  state.lastCondition = condition

  if (exportTrig === 1 && state.lastExport === 0) {
    const rows = state.list.map((v, i) => ({ index: i, value: typeof v === 'number' ? v.toFixed(6) : v }))
    const filename = `${params.filename || 'collected'}_${Date.now()}.csv`
    const columns = ['index', 'value']
    const escape = (v: unknown) => {
      const text = String(v ?? '')
      return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text
    }
    const csv = [columns.join(','), ...rows.map((r) => columns.map((c) => escape((r as any)[c])).join(','))].join('\n')
    downloadFile(filename, csv, 'text/csv')
  }
  state.lastExport = exportTrig

  return { list: [...state.list], count: state.list.length }
}

export const signalGate: NodeImpl = (inputs, params, ctx) => {
  let state: { last: unknown } = ctx.state.get(ctx.nodeId)
  if (!state) {
    state = { last: null }
    ctx.state.set(ctx.nodeId, state)
  }

  const value = inputs.value
  const gateRaw = inputs.gate
  const openWhen = Number(params.open_when) || 0
  const holdLast = !!params.hold_last

  const gate = typeof gateRaw === 'number' ? gateRaw : gateRaw === undefined ? 1 : Number(gateRaw) || 0
  const isOpen = openWhen === 0 ? gate > 0 : openWhen === 1 ? gate === 0 : true

  if (isOpen) {
    state.last = value
    return { out: value, passed: 1.0 }
  }
  return { out: holdLast ? state.last : null, passed: 0.0 }
}

export const logicLatch: NodeImpl = (inputs, params, ctx) => {
  let state: { held: unknown; prevReset: number } = ctx.state.get(ctx.nodeId)
  if (!state) {
    state = { held: null, prevReset: 0 }
    ctx.state.set(ctx.nodeId, state)
  }

  const value = inputs.value
  const reset = Number(inputs.reset) || 0
  const mode = Number(params.mode) || 0

  if (reset > 0.5 && state.prevReset <= 0.5) state.held = null
  state.prevReset = reset

  let shouldLatch: boolean
  if (mode === 0) shouldLatch = value !== null && value !== undefined
  else if (mode === 1) shouldLatch = value !== null && value !== undefined && Number(value) !== 0
  else shouldLatch = true

  if (shouldLatch) state.held = value

  return { held: state.held, active: state.held !== null && state.held !== undefined ? 1.0 : 0.0 }
}

export const logicCompare: NodeImpl = (inputs, params) => {
  const a = inputs.in_a ?? 0
  const b = inputs.in_b ?? 0
  const op = Number(params.op) || 0
  let result = false
  try {
    if (op === 0) result = a === b
    else if (op === 1) result = a !== b
    else if (op === 2) result = Number(a) > Number(b)
    else if (op === 3) result = Number(a) < Number(b)
    else if (op === 4) result = Number(a) >= Number(b)
    else if (op === 5) result = Number(a) <= Number(b)
  } catch {
    result = false
  }
  return { result }
}

export const logicPresence: NodeImpl = (inputs) => {
  const data = inputs.data
  if (data === null || data === undefined) return { found: false }
  if (Array.isArray(data)) return { found: data.length > 0 }
  if (typeof data === 'number') return { found: data !== 0 }
  return { found: true }
}

export const logicSwitch: NodeImpl = (inputs) => {
  const cond = typeof inputs.condition === 'number' ? inputs.condition !== 0 : !!inputs.condition
  return { output: cond ? inputs.if_true ?? null : inputs.if_false ?? null }
}

export const logicGateNode: NodeImpl = (inputs, params) => {
  const a = !!inputs.in_a
  const b = !!inputs.in_b
  const mode = Number(params.mode) || 0
  let result = false
  if (mode === 0) result = a && b
  else if (mode === 1) result = a || b
  else if (mode === 2) result = a !== b
  else if (mode === 3) result = !a
  return { result }
}

// --- strings --------------------------------------------------------------

export const stringConcat: NodeImpl = (inputs, params) => {
  const sep = String(params.separator ?? '')
  const list = inputs.list_in
  if (Array.isArray(list) && list.length > 0) return { result: list.map((x) => String(x)).join(sep) }
  const a = String(inputs.a ?? '')
  const b = String(inputs.b ?? '')
  return { result: `${a}${sep}${b}` }
}

export const stringSplit: NodeImpl = (inputs, params) => {
  const s = String(inputs.string ?? '')
  const sep = params.separator ?? ' '
  const parts = sep ? s.split(sep) : s.split('')
  return { list: parts, first: parts[0] ?? '' }
}

export const stringLength: NodeImpl = (inputs) => ({ length: String(inputs.string ?? '').length })

export const stringCase: NodeImpl = (inputs, params) => {
  const s = String(inputs.string ?? '')
  const mode = Number(params.mode) || 0
  return { result: mode === 0 ? s.toUpperCase() : s.toLowerCase() }
}

export const stringReplace: NodeImpl = (inputs, params) => {
  const s = String(inputs.string ?? '')
  const search = String(params.search ?? '')
  const replace = String(params.replace ?? '')
  const useRegex = !!params.use_regex
  const caseSensitive = params.case !== false
  if (!search) return { result: s }
  try {
    const flags = `g${caseSensitive ? '' : 'i'}`
    const pattern = useRegex ? search : search.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    return { result: s.replace(new RegExp(pattern, flags), replace) }
  } catch {
    return { result: s }
  }
}

export const stringInput: NodeImpl = (_inputs, params) => ({ result: String(params.value ?? '') })

// --- math -------------------------------------------------------------------

const MATH_CONTEXT_NAMES = [
  'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'sqrt', 'exp', 'log', 'log2', 'log10',
  'pow', 'floor', 'ceil', 'round', 'abs', 'min', 'max', 'pi', 'e', 'inf',
  'clamp', 'lerp', 'sign', 'frac', 'deg', 'rad',
]

function mathContextValues(): unknown[] {
  return [
    Math.sin, Math.cos, Math.tan, Math.asin, Math.acos, Math.atan, Math.atan2, Math.sqrt, Math.exp,
    Math.log, Math.log2, Math.log10, Math.pow, Math.floor, Math.ceil, Math.round, Math.abs, Math.min, Math.max,
    Math.PI, Math.E, Infinity,
    (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x)),
    (a: number, b: number, t: number) => a + (b - a) * t,
    (x: number) => (x > 0 ? 1 : x < 0 ? -1 : 0),
    (x: number) => x - Math.floor(x),
    (x: number) => (x * 180) / Math.PI,
    (x: number) => (x * Math.PI) / 180,
  ]
}

function extractPortIndex(key: string): number {
  const n = parseInt(key.split('_')[0], 10)
  return Number.isFinite(n) ? n : 9999
}

export const mathExpr: NodeImpl = (inputs, params, ctx) => {
  const expr = String(params.expression ?? 'a + b').trim()
  if (!expr) return { result: 0 }

  const sortedKeys = Object.keys(inputs).sort((a, b) => extractPortIndex(a) - extractPortIndex(b))
  const varNames: string[] = []
  const varValues: number[] = []
  sortedKeys.slice(0, 26).forEach((key, i) => {
    varNames.push(String.fromCharCode(97 + i))
    const val = inputs[key]
    const n = Number(val)
    varValues.push(val !== null && val !== undefined && Number.isFinite(n) ? n : 0)
  })

  try {
    const fn = new Function(...MATH_CONTEXT_NAMES, ...varNames, `"use strict"; return (${expr});`)
    const result = fn(...mathContextValues(), ...varValues)
    const n = Number(result)
    ctx.emit('error', '')
    return { result: Number.isFinite(n) ? n : 0 }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    ctx.emit('error', message)
    return { result: 0 }
  }
}

export const mathMapRange: NodeImpl = (inputs, params) => {
  const value = inputs.value
  if (value === null || value === undefined) return { result: 0 }
  const v = Number(value)
  if (!Number.isFinite(v)) return { result: 0 }

  const inMin = Number(params.in_min) || 0
  const inMax = params.in_max === undefined ? 1 : Number(params.in_max)
  const outMin = Number(params.out_min) || 0
  const outMax = params.out_max === undefined ? 100 : Number(params.out_max)
  const clamp = !!params.clamp

  const inRange = inMax - inMin
  if (Math.abs(inRange) < 1e-12) return { result: outMin }

  let t = (v - inMin) / inRange
  if (clamp) t = Math.max(0, Math.min(1, t))
  return { result: outMin + t * (outMax - outMin) }
}

export const utilFilterLabel: NodeImpl = (inputs, params, ctx) => {
  const items = inputs.list_in
  if (!Array.isArray(items)) return { list_out: [], item_out: null, labels_list: [] }

  const dicts = items.filter((i): i is Record<string, unknown> => i !== null && typeof i === 'object')
  const allLabels = [...new Set(dicts.map((i) => String(i.label ?? 'unknown')))].sort()

  const query = String(params.query ?? 'person').toLowerCase()
  const filtered = dicts.filter((i) => String(i.label ?? '').toLowerCase() === query)

  const display = allLabels.length > 0 ? `Found: ${allLabels.join(', ')}` : 'No labels found'
  ctx.emit('display_text', display)

  return { list_out: filtered, item_out: filtered[0] ?? null, labels_list: allLabels }
}

export const mathOperation: NodeImpl = (inputs, params) => {
  const a = typeof inputs.a === 'number' ? inputs.a : 0
  const b = typeof inputs.b === 'number' ? inputs.b : Number(params.value_b) || 0
  const op = Number(params.operation) || 0
  let result = 0
  if (op === 0) result = a + b
  else if (op === 1) result = a - b
  else if (op === 2) result = a * b
  else if (op === 3 && b !== 0) result = a / b
  const opChar = op === 0 ? '+' : op === 1 ? '-' : op === 2 ? '*' : '/'
  return { result, display_text: `${a.toFixed(2)} ${opChar} ${b.toFixed(2)} = ${result.toFixed(4)}` }
}
