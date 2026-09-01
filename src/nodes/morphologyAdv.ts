import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'

export const morphologyAdvNode: NodeDef = {
  typeId: 'feat_morphology_adv',
  label: 'Morphology (Advanced)',
  category: 'Segmentation',
  description: 'Opérations morphologiques : Opening enlève les grains isolés, Closing bouche les trous / rattache ce qui est proche.',
  inputs: [{ id: 'mask', label: 'mask', color: 'mask' }],
  outputs: [
    { id: 'main', label: 'image', color: 'image' },
    { id: 'mask', label: 'mask', color: 'mask' },
  ],
  params: [
    {
      id: 'operation',
      label: 'Operation',
      type: 'select',
      default: 0,
      options: [
        { label: 'Opening', value: 0 },
        { label: 'Closing', value: 1 },
        { label: 'Gradient', value: 2 },
        { label: 'Top Hat', value: 3 },
        { label: 'Black Hat', value: 4 },
        { label: 'Dilate', value: 5 },
        { label: 'Erode', value: 6 },
      ],
    },
    {
      id: 'shape',
      label: 'Kernel Shape',
      type: 'select',
      default: 0,
      options: [
        { label: 'Rect', value: 0 },
        { label: 'Cross', value: 1 },
        { label: 'Ellipse', value: 2 },
      ],
    },
    { id: 'size', label: 'Kernel Size', type: 'number', default: 5, min: 1, max: 31, step: 2 },
    { id: 'iterations', label: 'Iterations', type: 'number', default: 1, min: 1, max: 10, step: 1 },
  ],
  process(inputs, params, ctx) {
    const src = inputs.mask as any
    if (!src) return { main: undefined, mask: undefined }
    const cv = ctx.cv
    const shapeMap = [cv.MORPH_RECT, cv.MORPH_CROSS, cv.MORPH_ELLIPSE]
    const size = Number(params.size)
    const kernel = cv.getStructuringElement(shapeMap[Number(params.shape)], new cv.Size(size, size))
    const dst = trackMat(new cv.Mat())
    const anchor = new cv.Point(-1, -1)
    const iterations = Number(params.iterations)
    const opMap = [
      cv.MORPH_OPEN,
      cv.MORPH_CLOSE,
      cv.MORPH_GRADIENT,
      cv.MORPH_TOPHAT,
      cv.MORPH_BLACKHAT,
      null, // dilate
      null, // erode
    ]
    const opIdx = Number(params.operation)
    if (opIdx === 5) {
      cv.dilate(src, dst, kernel, anchor, iterations, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue())
    } else if (opIdx === 6) {
      cv.erode(src, dst, kernel, anchor, iterations, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue())
    } else {
      cv.morphologyEx(src, dst, opMap[opIdx], kernel, anchor, iterations, cv.BORDER_CONSTANT, cv.morphologyDefaultBorderValue())
    }
    kernel.delete()
    return { main: dst, mask: dst }
  },
}
