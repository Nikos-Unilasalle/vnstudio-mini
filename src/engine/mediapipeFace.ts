const TASKS_VISION_PKG = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14'
const TASKS_VISION_ESM_URL = `${TASKS_VISION_PKG}/+esm`
const WASM_URL = `${TASKS_VISION_PKG}/wasm`
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'

let landmarkerPromise: Promise<any> | null = null
let cachedNumFaces = -1

export async function getFaceLandmarker(numFaces: number): Promise<any> {
  if (landmarkerPromise && cachedNumFaces === numFaces) return landmarkerPromise

  cachedNumFaces = numFaces
  landmarkerPromise = (async () => {
    const mod: any = await import(/* @vite-ignore */ TASKS_VISION_ESM_URL)
    const { FaceLandmarker, FilesetResolver } = mod
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL)
    return FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL },
      runningMode: 'IMAGE',
      numFaces,
    })
  })()
  return landmarkerPromise
}
