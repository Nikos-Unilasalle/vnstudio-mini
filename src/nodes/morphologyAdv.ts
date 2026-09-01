import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'

export const morphologyAdvNode: NodeDef = {
  typeId: 'feat_morphology_adv',
  label: 'Morphology (Advanced)',
  category: 'Segmentation',
  description: 'Opening enlève les grains isolés, Closing bouche les trous / rattache ce qui est proche.',
  inputs: [{ id: 'main', label: 'mask', color: 'mask' }],
  outputs: [{ id: 'main', label: 'mask', color: 'mask' }],
  params: [
    {
      id: 'operation',
      label: 'Operation',
      type: 'select',
      default: 0,
      options: [
        { label: 'Opening', value: 0 },
        { label: 'Closing', value: 1 },
      ],
    },
    {
      id: 'shape',
      label: 'Shape',
      type: 'select',
      default: 2,
      options: [
        { label: 'Rect', value: 0 },
        { label: 'Cross', value: 1 },
        { label: 'Ellipse', value: 2 },
      ],
    },
    { id: 'size', label: 'Kernel size', type: 'number', default: 7, min: 1, max: 99, step: 2 },
    { id: 'iterations', label: 'Iterations', type: 'number', default: 1, min: 1, max: 10, step: 1 },
  ],
  process(inputs, params, ctx) {
    const src = inputs.main as any
    if (!src) return { main: undefined }
    const cv = ctx.cv
    const shapeMap = [cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE]
    const size = Number(params.size)
    const kernel = cv.getStructuringElement(shapeMap[Number(params.shape)], new cv.Size(size, size))
    const dst = trackMat(new cv.Mat())
    const op = Number(params.operation) === 0 ? cv.MORPH_OPEN : cv.MORPH_CLOSE
    const anchor = new cv.Point(-1, -1)
    cv.morphologyEx(src, dst, op, kernel, anchor, Number(params.iterations), cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue())
    kernel.delete()
    return { main: dst }
  },
}
