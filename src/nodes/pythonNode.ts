import type { NodeDef } from '../engine/types'

const DEFAULT_CODE = 'out_a = a\nout_b = null\nout_c = null'

export const pythonNode: NodeDef = {
  typeId: 'logic_python',
  label: 'Python Node',
  category: 'Logic',
  description:
    "Un script pour ce qu'aucune node ne fait. Écrit du JavaScript ici (le moteur web n'embarque pas Python) : toute variable out_a/out_b/out_c devient un port de sortie. Entrées disponibles: a, b, c.",
  inputs: [
    { id: 'a', label: 'a', color: 'dict' },
    { id: 'b', label: 'b', color: 'dict' },
    { id: 'c', label: 'c', color: 'dict' },
  ],
  outputs: [
    { id: 'out_a', label: 'out_a', color: 'dict' },
    { id: 'out_b', label: 'out_b', color: 'dict' },
    { id: 'out_c', label: 'out_c', color: 'dict' },
  ],
  params: [{ id: 'code', label: 'Code (JS)', type: 'string', default: DEFAULT_CODE }],
  process(inputs, params) {
    const code = String(params.code ?? DEFAULT_CODE)
    try {
      const fn = new Function(
        'a',
        'b',
        'c',
        `"use strict";\nlet out_a, out_b, out_c;\n${code}\nreturn { out_a, out_b, out_c };`
      )
      const result = fn(inputs.a ?? null, inputs.b ?? null, inputs.c ?? null)
      return { out_a: result.out_a ?? null, out_b: result.out_b ?? null, out_c: result.out_c ?? null, __error: undefined } as any
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      return { out_a: null, out_b: null, out_c: null, __text: `Erreur: ${message}` } as any
    }
  },
}
