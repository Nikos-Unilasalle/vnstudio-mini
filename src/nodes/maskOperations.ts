import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'

export const maskOperationsNode: NodeDef = {
  typeId: 'mask_operations',
  label: 'Mask Operations',
  category: 'Mask',
  description: 'Opération bit-à-bit entre deux masques : addition (OR), soustraction (A - B), intersection (AND).',
  inputs: [
    { id: 'mask_a', label: 'mask A', color: 'mask' },
    { id: 'mask_b', label: 'mask B', color: 'mask' },
  ],
  outputs: [{ id: 'mask', label: 'mask', color: 'mask' }],
  params: [
    {
      id: 'operation',
      label: 'Operation',
      type: 'select',
      default: 0,
      options: [
        { label: 'Addition (OR)', value: 0 },
        { label: 'Subtraction (A - B)', value: 1 },
        { label: 'Intersection (AND)', value: 2 },
      ],
    },
  ],
  process(inputs, params, ctx) {
    const a = inputs.mask_a as any
    const b = inputs.mask_b as any
    if (!a && !b) return { mask: undefined }
    const cv = ctx.cv
    const ref = a ?? b

    const m1 = trackMat(new cv.Mat())
    if (a) {
      if (a.channels() === 1) a.copyTo(m1)
      else cv.cvtColor(a, m1, cv.COLOR_BGR2GRAY)
    } else {
      new cv.Mat.zeros(ref.rows, ref.cols, cv.CV_8U).copyTo(m1)
    }

    const m2 = trackMat(new cv.Mat())
    if (b) {
      if (b.channels() === 1) b.copyTo(m2)
      else cv.cvtColor(b, m2, cv.COLOR_BGR2GRAY)
    } else {
      new cv.Mat.zeros(ref.rows, ref.cols, cv.CV_8U).copyTo(m2)
    }

    const op = Number(params.operation)
    const res = trackMat(new cv.Mat())
    if (op === 0) {
      cv.bitwise_or(m1, m2, res)
    } else if (op === 1) {
      const notB = trackMat(new cv.Mat())
      cv.bitwise_not(m2, notB)
      cv.bitwise_and(m1, notB, res)
    } else {
      cv.bitwise_and(m1, m2, res)
    }
    return { mask: res }
  },
}
