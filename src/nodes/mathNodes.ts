import type { NodeDef } from '../engine/types'

function binaryMath(
  typeId: string,
  label: string,
  defaultA: number,
  defaultB: number,
  fn: (a: number, b: number) => number
): NodeDef {
  return {
    typeId,
    label,
    category: 'Math',
    inputs: [
      { id: 'a', label: 'a', color: 'scalar' },
      { id: 'b', label: 'b', color: 'scalar' },
    ],
    outputs: [{ id: 'result', label: 'result', color: 'scalar' }],
    params: [
      { id: 'value_a', label: 'A (si déconnecté)', type: 'number', default: defaultA, step: 0.1 },
      { id: 'value_b', label: 'B (si déconnecté)', type: 'number', default: defaultB, step: 0.1 },
    ],
    process(inputs, params) {
      const a = typeof inputs.a === 'number' ? inputs.a : Number(params.value_a)
      const b = typeof inputs.b === 'number' ? inputs.b : Number(params.value_b)
      return { result: fn(a, b) }
    },
  }
}

function unaryMath(typeId: string, label: string, defaultA: number, fn: (a: number) => number): NodeDef {
  return {
    typeId,
    label,
    category: 'Math',
    inputs: [{ id: 'a', label: 'a', color: 'scalar' }],
    outputs: [{ id: 'result', label: 'result', color: 'scalar' }],
    params: [{ id: 'value_a', label: 'Value', type: 'number', default: defaultA, step: 0.1 }],
    process(inputs, params) {
      const a = typeof inputs.a === 'number' ? inputs.a : Number(params.value_a)
      return { result: fn(a) }
    },
  }
}

export const mathAddNode = binaryMath('math_add', 'Math: Add', 0, 0, (a, b) => a + b)
export const mathSubNode = binaryMath('math_sub', 'Math: Subtract', 0, 0, (a, b) => a - b)
export const mathMulNode = binaryMath('math_mul', 'Math: Multiply', 1, 1, (a, b) => a * b)
export const mathDivNode = binaryMath('math_div', 'Math: Divide', 1, 1, (a, b) => (b !== 0 ? a / b : 0))
export const mathModNode = binaryMath('math_mod', 'Math: Modulo', 0, 1, (a, b) => (b !== 0 ? a % b : 0))
export const mathAbsNode = unaryMath('math_abs', 'Math: Absolute', 0, Math.abs)
export const mathRoundNode = unaryMath('math_round', 'Math: Round', 0, Math.round)

export const mathDistanceNode: NodeDef = {
  typeId: 'math_distance',
  label: 'Math: Distance',
  category: 'Math',
  description: 'Distance euclidienne entre deux points (dict avec x/y).',
  inputs: [
    { id: 'p1', label: 'p1', color: 'dict' },
    { id: 'p2', label: 'p2', color: 'dict' },
  ],
  outputs: [{ id: 'result', label: 'result', color: 'scalar' }],
  params: [],
  process(inputs) {
    const p1 = (inputs.p1 as Record<string, number>) ?? {}
    const p2 = (inputs.p2 as Record<string, number>) ?? {}
    const x1 = p1.x ?? p1.xmin ?? 0
    const y1 = p1.y ?? p1.ymin ?? 0
    const x2 = p2.x ?? p2.xmin ?? 0
    const y2 = p2.y ?? p2.ymin ?? 0
    return { result: Math.hypot(x2 - x1, y2 - y1) }
  },
}

export const numberNode: NodeDef = {
  typeId: 'scalar_input',
  label: 'Number',
  category: 'Math',
  description: 'Constante numérique réglable.',
  inputs: [],
  outputs: [{ id: 'value', label: 'value', color: 'scalar' }],
  params: [{ id: 'value', label: 'Value', type: 'number', default: 0, step: 0.1 }],
  process(_inputs, params) {
    return { value: Number(params.value) }
  },
}
