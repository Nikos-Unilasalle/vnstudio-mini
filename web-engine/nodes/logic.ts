import type { NodeImpl } from '../types'

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
