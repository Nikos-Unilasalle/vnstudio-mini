import type { NodeDef } from '../engine/types'
import { getFaceLandmarker } from '../engine/mediapipeFace'

export const faceTrackerNode: NodeDef = {
  typeId: 'analysis_face_mp',
  label: 'Face Tracker',
  category: 'Body',
  description: 'Détecte et suit un ou plusieurs visages et leurs points de repère (MediaPipe, 478 points dont iris).',
  inputs: [{ id: 'image', label: 'image', color: 'image' }],
  outputs: [
    { id: 'faces_list', label: 'faces_list', color: 'list' },
    { id: 'main', label: 'image', color: 'image' },
    { id: 'face_0', label: 'face_0', color: 'dict' },
  ],
  params: [{ id: 'max_faces', label: 'Max Faces', type: 'number', default: 1, min: 1, max: 10, step: 1 }],
  async process(inputs, params, ctx) {
    const image = inputs.image as any
    if (!image) return { faces_list: [], main: undefined, face_0: null }
    const cv = ctx.cv
    const maxFaces = Number(params.max_faces)

    const canvas = document.createElement('canvas')
    canvas.width = image.cols
    canvas.height = image.rows
    const rgba = new cv.Mat()
    cv.cvtColor(image, rgba, cv.COLOR_BGR2RGBA)
    cv.imshow(canvas, rgba)
    rgba.delete()

    const landmarker = await getFaceLandmarker(maxFaces)
    const result = landmarker.detect(canvas)

    const facesList: any[] = []
    const rawFaces = result.faceLandmarks ?? []
    for (const lms of rawFaces) {
      const xs = lms.map((p: any) => p.x)
      const ys = lms.map((p: any) => p.y)
      const xmin = Math.max(0, Math.min(...xs))
      const ymin = Math.max(0, Math.min(...ys))
      const xmax = Math.max(...xs)
      const ymax = Math.max(...ys)
      facesList.push({
        xmin,
        ymin,
        width: Math.min(1 - xmin, xmax - xmin),
        height: Math.min(1 - ymin, ymax - ymin),
        landmarks: lms.map((p: any) => ({ x: p.x, y: p.y, z: p.z })),
        label: 'face',
      })
    }

    const out: Record<string, unknown> = { faces_list: facesList, main: image }
    facesList.forEach((f, i) => {
      out[`face_${i}`] = f
    })
    if (!('face_0' in out)) out.face_0 = null
    return out
  },
}
