import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'

export const thresholdAdvNode: NodeDef = {
  typeId: 'feat_threshold_adv',
  label: 'Threshold (Advanced)',
  category: 'Segmentation',
  description: "Sépare clair et sombre. Otsu = seuil automatique, Binary = seuil manuel.",
  inputs: [{ id: 'main', label: 'image', color: 'image' }],
  outputs: [{ id: 'main', label: 'mask', color: 'mask' }],
  params: [
    {
      id: 'mode',
      label: 'Mode',
      type: 'select',
      default: 0,
      options: [
        { label: 'Otsu', value: 0 },
        { label: 'Otsu Inv', value: 1 },
        { label: 'Binary', value: 2 },
      ],
    },
    { id: 'threshold', label: 'Threshold (Binary)', type: 'number', default: 127, min: 0, max: 255, step: 1 },
  ],
  process(inputs, params, ctx) {
    const src = inputs.main as any
    if (!src) return { main: undefined }
    const cv = ctx.cv
    const gray = trackMat(new cv.Mat())
    if (src.channels() === 1) src.copyTo(gray)
    else cv.cvtColor(src, gray, cv.COLOR_BGR2GRAY)

    const dst = trackMat(new cv.Mat())
    const mode = Number(params.mode)
    if (mode === 0) {
      cv.threshold(gray, dst, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    } else if (mode === 1) {
      cv.threshold(gray, dst, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    } else {
      cv.threshold(gray, dst, Number(params.threshold), 255, cv.THRESH_BINARY)
    }
    return { main: dst }
  },
}
