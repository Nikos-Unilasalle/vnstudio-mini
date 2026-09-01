import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'

export const connectedCompCv2Node: NodeDef = {
  typeId: 'sci_connected_components_cv2',
  label: 'Connected Comp. (CV2)',
  category: 'Segmentation',
  description: "Comme Connected Components, mais sort des markers bruts — la seule entrée que le Watershed accepte.",
  inputs: [{ id: 'main', label: 'mask', color: 'mask' }],
  outputs: [{ id: 'markers', label: 'markers', color: 'regions' }],
  params: [
    {
      id: 'connectivity',
      label: 'Connectivity',
      type: 'select',
      default: 0,
      options: [
        { label: '8-connexe', value: 0 },
        { label: '4-connexe', value: 1 },
      ],
    },
  ],
  process(inputs, params, ctx) {
    const src = inputs.main as any
    if (!src) return { markers: undefined }
    const cv = ctx.cv
    const labels = trackMat(new cv.Mat())
    const connectivity = Number(params.connectivity) === 1 ? 4 : 8
    cv.connectedComponents(src, labels, connectivity, cv.CV_32S)
    return { markers: labels }
  },
}
