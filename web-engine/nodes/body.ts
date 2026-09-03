import type { NodeImpl } from '../types'
import { getFaceLandmarker, getHandLandmarker, getObjectDetector, getPoseLandmarker } from '../mediapipe'
import { toBgr } from '../cvUtils'
import { makeCanvas, drawMatToCanvas } from '../canvasCompat'

/** MediaPipe wants an image-like source, not a Mat, so round-trip via a canvas. */
function matToCanvasRgba(cv: any, mat: any): OffscreenCanvas {
  const canvas = makeCanvas(mat.cols, mat.rows)
  const rgba = new cv.Mat()
  cv.cvtColor(mat, rgba, mat.channels() === 1 ? cv.COLOR_GRAY2RGBA : cv.COLOR_BGR2RGBA)
  drawMatToCanvas(cv, canvas, rgba)
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

export const analysisPoseMp: NodeImpl = async (inputs, _params, ctx) => {
  const image = inputs.image as any
  if (!image) return { main: null, pose_list: [], data: {} }

  const canvas = matToCanvasRgba(ctx.cv, image)
  const landmarker = await getPoseLandmarker(1)
  const result = landmarker.detect(canvas)
  const detected: { x: number; y: number; z: number; visibility?: number }[][] = result.landmarks ?? []

  const poses = detected.map((landmarks) => ({
    ...boundsOf(landmarks),
    landmarks: landmarks.map((p) => ({ x: p.x, y: p.y, z: p.z, visibility: p.visibility ?? 1 })),
    label: 'pose',
  }))

  const overlay = ctx.track(drawLandmarks(ctx.cv, image, detected, [255, 255, 0]))

  return { main: overlay, pose_list: poses, data: poses[0] ?? {} }
}

function drawBoxes(cv: any, image: any, boxes: { xmin: number; ymin: number; width: number; height: number; label: string; score: number }[]): any {
  const overlay = toBgr(cv, image)
  const w = overlay.cols
  const h = overlay.rows
  const scalar = new cv.Scalar(255, 0, 255, 255)
  for (const box of boxes) {
    const x1 = Math.round(box.xmin * w)
    const y1 = Math.round(box.ymin * h)
    const x2 = Math.round((box.xmin + box.width) * w)
    const y2 = Math.round((box.ymin + box.height) * h)
    cv.rectangle(overlay, new cv.Point(x1, y1), new cv.Point(x2, y2), scalar, 2)
    cv.putText(overlay, `${box.label} ${box.score.toFixed(2)}`, new cv.Point(x1, Math.max(10, y1 - 6)), cv.FONT_HERSHEY_SIMPLEX, 0.5, scalar, 1, cv.LINE_AA)
  }
  return overlay
}

export const analysisObjectMp: NodeImpl = async (inputs, params, ctx) => {
  const image = inputs.image as any
  if (!image) return { main: null, objects_list: [] }

  const scoreThreshold = (Number(params.score_threshold) || 50) / 100
  const maxResults = Math.max(1, Math.round(Number(params.max_results) || 5))

  const canvas = matToCanvasRgba(ctx.cv, image)
  const detector = await getObjectDetector(scoreThreshold, maxResults)
  const result = detector.detect(canvas)
  const detections: { boundingBox: { originX: number; originY: number; width: number; height: number }; categories: { categoryName: string; score: number }[] }[] = result.detections ?? []

  const w = canvas.width
  const h = canvas.height
  const objects = detections.map((d) => {
    const bbox = d.boundingBox
    const category = d.categories[0]
    const xmin = bbox.originX / w
    const ymin = bbox.originY / h
    const width = bbox.width / w
    const height = bbox.height / h
    return {
      label: category.categoryName,
      score: category.score,
      xmin,
      ymin,
      width,
      height,
      _type: 'graphics' as const,
      shape: 'rect' as const,
      pts: [
        [xmin, ymin],
        [xmin + width, ymin + height],
      ],
      r: 255,
      g: 0,
      b: 255,
      thickness: 2,
    }
  })

  const overlay = ctx.track(drawBoxes(ctx.cv, image, objects))

  const outputs: Record<string, unknown> = { main: overlay, objects_list: objects }
  for (let i = 0; i < 5; i++) outputs[`obj_${i}`] = objects[i] ?? null
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
