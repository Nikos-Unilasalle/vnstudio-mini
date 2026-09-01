import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'

export const thresholdAdvNode: NodeDef = {
  typeId: 'feat_threshold_adv',
  label: 'Threshold (Advanced)',
  category: 'Segmentation',
  description: 'Seuillage avancé : binaire manuel, Otsu (auto), ou pourcentage du maximum.',
  inputs: [{ id: 'image', label: 'image', color: 'image' }],
  outputs: [
    { id: 'main', label: 'image', color: 'image' },
    { id: 'mask', label: 'mask', color: 'mask' },
  ],
  params: [
    {
      id: 'mode',
      label: 'Mode',
      type: 'select',
      default: 0,
      options: [
        { label: 'Binary', value: 0 },
        { label: 'Binary Inv', value: 1 },
        { label: 'Otsu', value: 2 },
        { label: 'Otsu Inv', value: 3 },
        { label: '70% of Max', value: 4 },
      ],
    },
    { id: 'threshold', label: 'Value', type: 'number', default: 127, min: 0, max: 255, step: 1 },
  ],
  process(inputs, params, ctx) {
    const src = inputs.image as any
    if (!src) return { main: undefined, mask: undefined }
    const cv = ctx.cv
    const gray = trackMat(new cv.Mat())
    if (src.channels() === 1) src.copyTo(gray)
    else cv.cvtColor(src, gray, cv.COLOR_BGR2GRAY)

    const dst = trackMat(new cv.Mat())
    const mode = Number(params.mode)
    const val = Number(params.threshold)
    if (mode === 0) {
      cv.threshold(gray, dst, val, 255, cv.THRESH_BINARY)
    } else if (mode === 1) {
      cv.threshold(gray, dst, val, 255, cv.THRESH_BINARY_INV)
    } else if (mode === 2) {
      cv.threshold(gray, dst, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    } else if (mode === 3) {
      cv.threshold(gray, dst, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    } else {
      let maxVal = 0
      const data = gray.data as Uint8Array
      for (let i = 0; i < data.length; i++) if (data[i] > maxVal) maxVal = data[i]
      cv.threshold(gray, dst, 0.7 * maxVal, 255, cv.THRESH_BINARY)
    }
    return { main: dst, mask: dst }
  },
}
