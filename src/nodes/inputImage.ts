import type { NodeDef } from '../engine/types'
import { loadImageAsMat } from '../engine/imageIO'
import { trackMat } from '../engine/executor'

export const inputImageNode: NodeDef = {
  typeId: 'input_image',
  label: 'Image File',
  category: 'Input',
  description: "Charge une image (échantillon fourni ou fichier importé) dans le graphe.",
  inputs: [],
  outputs: [{ id: 'main', label: 'image', color: 'image' }],
  params: [
    {
      id: 'source',
      label: 'Source',
      type: 'select',
      default: 'samples/galets_atelier.jpg',
      options: [
        { label: 'galets_atelier.jpg', value: 'samples/galets_atelier.jpg' },
        { label: 'berge_meuse.jpg', value: 'samples/berge_meuse.jpg' },
        { label: 'empreinte_atelier.jpg', value: 'samples/empreinte_atelier.jpg' },
        { label: 'empreinte_scene.jpg', value: 'samples/empreinte_scene.jpg' },
        { label: 'Fichier importé…', value: '__upload__' },
      ],
    },
    { id: 'uploadedDataUrl', label: 'Fichier importé', type: 'file', default: '' },
  ],
  async process(_inputs, params, ctx) {
    const source = params.source as string
    const src = source === '__upload__' ? (params.uploadedDataUrl as string) : `${import.meta.env.BASE_URL}${source}`
    if (!src) return { main: undefined }
    const mat = trackMat(await loadImageAsMat(ctx.cv, src))
    return { main: mat }
  },
}
