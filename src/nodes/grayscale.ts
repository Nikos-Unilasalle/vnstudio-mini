import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'

export const grayscaleNode: NodeDef = {
  typeId: 'filter_gray',
  label: 'Grayscale',
  category: 'Image',
  description: 'Passe en niveaux de gris.',
  inputs: [{ id: 'main', label: 'image', color: 'image' }],
  outputs: [{ id: 'main', label: 'image', color: 'image' }],
  params: [],
  process(inputs, _params, ctx) {
    const src = inputs.main as any
    if (!src) return { main: undefined }
    const dst = trackMat(new ctx.cv.Mat())
    if (src.channels() === 1) {
      src.copyTo(dst)
    } else {
      ctx.cv.cvtColor(src, dst, ctx.cv.COLOR_BGR2GRAY)
    }
    return { main: dst }
  },
}
