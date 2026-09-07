/**
 * MediaPipe Tasks (Vision) loader.
 *
 * The desktop engine imports mediapipe from the Python venv and downloads the
 * .task bundles to disk. In the browser both come from a CDN: the ESM build of
 * @mediapipe/tasks-vision, and the same float16 model files Google publishes.
 * Nothing is bundled, so the initial app download stays small — the model is
 * only fetched when a tracker node actually runs.
 */
const TASKS_VISION_PKG = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14'
const TASKS_VISION_ESM = `${TASKS_VISION_PKG}/+esm`
const WASM_ROOT = `${TASKS_VISION_PKG}/wasm`

const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
const HAND_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
const POSE_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
const OBJECT_MODEL = 'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite'

let visionModule: Promise<any> | null = null

function loadVisionModule(): Promise<any> {
  if (!visionModule) visionModule = import(/* @vite-ignore */ TASKS_VISION_ESM)
  return visionModule
}

const detectors = new Map<string, Promise<any>>()

/**
 * Last timestamp handed to each detector.
 *
 * The VIDEO running mode tracks landmarks from one frame to the next instead of
 * detecting from scratch, which is what a webcam graph wants — but it demands
 * strictly increasing timestamps, and it throws if two calls share one. A graph
 * can hold two nodes that resolve to the same cached detector, so the counter
 * lives with the detector rather than with the caller.
 */
const timestamps = new WeakMap<object, number>()

/**
 * Runs a detector in VIDEO mode on `canvas`, with a timestamp guaranteed to be
 * ahead of the previous one for that detector.
 */
export function detectOnVideo(detector: any, canvas: unknown): any {
  const previous = timestamps.get(detector) ?? 0
  const now = Math.max(previous + 1, Math.round(performance.now()))
  timestamps.set(detector, now)
  return detector.detectForVideo(canvas, now)
}

export async function getFaceLandmarker(numFaces: number): Promise<any> {
  const key = `face:${numFaces}`
  let pending = detectors.get(key)
  if (pending) return pending

  pending = (async () => {
    const { FaceLandmarker, FilesetResolver } = await loadVisionModule()
    const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT)
    return FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: FACE_MODEL },
      runningMode: 'VIDEO',
      numFaces,
    })
  })()
  detectors.set(key, pending)
  return pending
}

export async function getHandLandmarker(numHands: number): Promise<any> {
  const key = `hand:${numHands}`
  let pending = detectors.get(key)
  if (pending) return pending

  pending = (async () => {
    const { HandLandmarker, FilesetResolver } = await loadVisionModule()
    const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT)
    return HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: HAND_MODEL },
      runningMode: 'VIDEO',
      numHands,
    })
  })()
  detectors.set(key, pending)
  return pending
}

export async function getPoseLandmarker(numPoses: number): Promise<any> {
  const key = `pose:${numPoses}`
  let pending = detectors.get(key)
  if (pending) return pending

  pending = (async () => {
    const { PoseLandmarker, FilesetResolver } = await loadVisionModule()
    const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT)
    return PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: POSE_MODEL },
      runningMode: 'VIDEO',
      numPoses,
    })
  })()
  detectors.set(key, pending)
  return pending
}

export async function getObjectDetector(scoreThreshold: number, maxResults: number): Promise<any> {
  const key = `object:${scoreThreshold}:${maxResults}`
  let pending = detectors.get(key)
  if (pending) return pending

  pending = (async () => {
    const { ObjectDetector, FilesetResolver } = await loadVisionModule()
    const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT)
    return ObjectDetector.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: OBJECT_MODEL },
      runningMode: 'VIDEO',
      scoreThreshold,
      maxResults,
    })
  })()
  detectors.set(key, pending)
  return pending
}
