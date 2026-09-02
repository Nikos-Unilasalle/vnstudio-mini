import type { NodeImpl } from '../types'
import { getFaceLandmarker, getHandLandmarker } from '../mediapipe'
import { toBgr } from '../cvUtils'

/** MediaPipe wants an HTML element or ImageData, not a Mat, so round-trip via a canvas. */
function matToCanvasRgba(cv: any, mat: any): HTMLCanvasElement {
  const canvas = document.createElement('canvas')
  canvas.width = mat.cols
  canvas.height = mat.rows
  const rgba = new cv.Mat()
  cv.cvtColor(mat, rgba, mat.channels() === 1 ? cv.COLOR_GRAY2RGBA : cv.COLOR_BGR2RGBA)
  cv.imshow(canvas, rgba)
  rgba.delete()
  return canvas
}

function boundsOf(landmarks: { x: number; y: number }[]) {
  const xs = landmarks.map((p) => p.x)
  const ys = landmarks.map((p) => p.y)
  const xmin = Math.max(0, Math.min(...xs))
  const ymin = Math.max(0, Math.min(...ys))
  return {
    xmin,
    ymin,
    width: Math.min(1 - xmin, Math.max(...xs) - xmin),
    height: Math.min(1 - ymin, Math.max(...ys) - ymin),
  }
}

function drawLandmarks(cv: any, image: any, groups: { x: number; y: number }[][], colour: [number, number, number]): any {
  const overlay = toBgr(cv, image)
  const radius = Math.max(1, Math.round(Math.max(overlay.cols, overlay.rows) / 500))
  const scalar = new cv.Scalar(colour[2], colour[1], colour[0], 255)
  for (const landmarks of groups) {
    for (const point of landmarks) {
      cv.circle(overlay, new cv.Point(Math.round(point.x * overlay.cols), Math.round(point.y * overlay.rows)), radius, scalar, -1)
    }
  }
  return overlay
}

export const analysisFaceMp: NodeImpl = async (inputs, params, ctx) => {
  const image = inputs.image as any
  if (!image) return { faces_list: [], main: null }

  const canvas = matToCanvasRgba(ctx.cv, image)
  const landmarker = await getFaceLandmarker(Math.max(1, Number(params.max_faces) || 3))
  const result = landmarker.detect(canvas)
  const detected: { x: number; y: number; z: number }[][] = result.faceLandmarks ?? []

  const faces = detected.map((landmarks) => ({
    ...boundsOf(landmarks),
    landmarks: landmarks.map((p) => ({ x: p.x, y: p.y, z: p.z })),
    label: 'face',
  }))

  const overlay = ctx.track(drawLandmarks(ctx.cv, image, detected, [0, 255, 0]))
  ctx.emit('count', faces.length)

  const outputs: Record<string, unknown> = { faces_list: faces, main: overlay }
  // The desktop node exposes each detection on its own port, which is how a
  // Point Tracker downstream picks "face 0" without a List Selector in between.
  faces.forEach((face, i) => {
    outputs[`face_${i}`] = face
  })
  return outputs
}

export const analysisHandMp: NodeImpl = async (inputs, params, ctx) => {
  const image = inputs.image as any
  if (!image) return { hands_list: [], main: null }

  const canvas = matToCanvasRgba(ctx.cv, image)
  const landmarker = await getHandLandmarker(Math.max(1, Number(params.max_hands) || 2))
  const result = landmarker.detect(canvas)
  const detected: { x: number; y: number; z: number }[][] = result.landmarks ?? []

  const hands = detected.map((landmarks, i) => ({
    ...boundsOf(landmarks),
    landmarks: landmarks.map((p) => ({ x: p.x, y: p.y, z: p.z })),
    label: result.handedness?.[i]?.[0]?.categoryName ?? 'hand',
  }))

  const overlay = ctx.track(drawLandmarks(ctx.cv, image, detected, [255, 128, 0]))
  ctx.emit('count', hands.length)

  const outputs: Record<string, unknown> = { hands_list: hands, main: overlay }
  hands.forEach((hand, i) => {
    outputs[`hand_${i}`] = hand
  })
  return outputs
}

export const geomTrackPoint: NodeImpl = (inputs, params) => {
  const data = inputs.data as { landmarks?: { x: number; y: number }[] } | null | undefined
  const empty = { x: 0, y: 0, draw: null }
  if (!data || !Array.isArray(data.landmarks)) return empty

  const pointId = Number(params.point_id) || 0
  const landmarks = data.landmarks
  if (pointId < 0 || pointId >= landmarks.length) return empty

  const landmark = landmarks[pointId]
  const absolute = !!params.absolute
  let scaleX = 1
  let scaleY = 1
  if (absolute) {
    const image = inputs.image as any
    // Landmarks are normalised per-axis: x by width, y by height. Without the
    // image to scale by, any oblique distance computed downstream is wrong.
    scaleX = image ? image.cols : 640
    scaleY = image ? image.rows : 480
  }

  return {
    x: landmark.x * scaleX,
    y: landmark.y * scaleY,
    draw: {
      shape: 'point',
      pts: [[landmark.x * scaleX, landmark.y * scaleY]],
      relative: !absolute,
      thickness: Number(params.thickness) || 5,
      r: Number(params.r) || 0,
      g: Number(params.g) ?? 255,
      b: Number(params.b) || 0,
    },
  }
}
