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
