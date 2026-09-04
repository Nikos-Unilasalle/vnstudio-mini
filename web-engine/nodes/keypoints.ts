import type { NodeImpl, RunContext } from '../types'
import { toBgr, toGray } from '../cvUtils'

/**
 * The dict shape the desktop detectors emit for every keypoint — a graphics
 * point that Draw Overlay can render directly, carrying the extra keypoint
 * fields that Feature Matcher needs to rebuild pixel coordinates.
 */
interface KeypointDict {
  _type: 'graphics'
  shape: 'point'
  pts: [number, number][]
  relative: true
  color: string
  size?: number
  angle?: number
  response?: number
}

function keypointsToDicts(vector: any, width: number, height: number, colour: string, withDetail: boolean): KeypointDict[] {
  const out: KeypointDict[] = []
  for (let i = 0; i < vector.size(); i++) {
    const k = vector.get(i)
    const dict: KeypointDict = {
      _type: 'graphics',
      shape: 'point',
      pts: [[k.pt.x / width, k.pt.y / height]],
      relative: true,
      color: colour,
    }
    if (withDetail) {
      dict.size = k.size
      dict.angle = k.angle
      dict.response = k.response
    }
    out.push(dict)
  }
  return out
}

/**
 * Detectors are expensive to construct, so one is kept per node and rebuilt only
 * when a parameter that feeds the constructor changes.
 */
function cachedDetector(ctx: RunContext, signature: string, build: () => any): any {
  const state = ctx.state.get(ctx.nodeId) as { signature: string; detector: any } | undefined
  if (state && state.signature === signature) return state.detector
  state?.detector?.delete?.()
  const detector = build()
  ctx.state.set(ctx.nodeId, { signature, detector })
  return detector
}

export const featOrb: NodeImpl = (inputs, params, ctx) => {
  const img = (inputs.image ?? inputs.main) as any
  if (!img) return { keypoints: [], descriptors: null }
  const cv = ctx.cv

  const nFeatures = Math.max(10, Math.round(Number(params.n_features) || 500))
  const scaleFactor = Number(params.scale_factor) || 1.2
  const nLevels = Math.max(1, Math.round(Number(params.n_levels) || 8))

  const orb = cachedDetector(ctx, `orb:${nFeatures}:${scaleFactor}:${nLevels}`, () => new cv.ORB(nFeatures, scaleFactor, nLevels))

  const gray = toGray(cv, img)
  const keypoints = new cv.KeyPointVector()
  const descriptors = ctx.track(new cv.Mat())
  const noMask = new cv.Mat()
  orb.detectAndCompute(gray, noMask, keypoints, descriptors)

  const dicts = keypointsToDicts(keypoints, gray.cols, gray.rows, '#00ffff', true)
  keypoints.delete()
  noMask.delete()
  gray.delete()

  ctx.emit('count', dicts.length)
  return { keypoints: dicts, descriptors }
}

/**
 * The desktop node runs SIFT, which this OpenCV build does not ship — SIFT lives
 * in opencv_contrib and is compiled out. AKAZE is the closest thing available:
 * scale- and rotation-invariant, non-linear scale space, free to use. The
 * `detector` output says which one actually ran so the substitution is visible
 * rather than silent.
 */
export const featSift: NodeImpl = (inputs, params, ctx) => {
  const img = (inputs.image ?? inputs.main) as any
  if (!img) return { keypoints: [], descriptors: null, detector: '' }
  const cv = ctx.cv

  const akaze = cachedDetector(ctx, 'akaze', () => new cv.AKAZE())

  const gray = toGray(cv, img)
  const keypoints = new cv.KeyPointVector()
  const descriptors = ctx.track(new cv.Mat())
  const noMask = new cv.Mat()
  akaze.detectAndCompute(gray, noMask, keypoints, descriptors)

  let dicts = keypointsToDicts(keypoints, gray.cols, gray.rows, '#ffff00', true)
  // AKAZE has no feature cap of its own, so honour the node's by keeping the
  // strongest responses — the same subset SIFT's nfeatures would have kept.
  const limit = Math.round(Number(params.n_features) || 0)
  if (limit > 0 && dicts.length > limit) {
    dicts = [...dicts].sort((a, b) => (b.response ?? 0) - (a.response ?? 0)).slice(0, limit)
  }

  keypoints.delete()
  noMask.delete()
  gray.delete()

  ctx.emit('count', dicts.length)
  return { keypoints: dicts, descriptors, detector: 'AKAZE (SIFT is not in this OpenCV build)' }
}

export const featFast: NodeImpl = (inputs, params, ctx) => {
  const img = (inputs.image ?? inputs.main) as any
  if (!img) return { keypoints: [] }
  const cv = ctx.cv

  const threshold = Math.max(1, Math.round(Number(params.threshold) || 10))
  const nonmax = params.nonmax !== false
  const fast = cachedDetector(ctx, `fast:${threshold}:${nonmax}`, () => new cv.FastFeatureDetector(threshold, nonmax))

  const gray = toGray(cv, img)
  const keypoints = new cv.KeyPointVector()
  fast.detect(gray, keypoints)

  const dicts = keypointsToDicts(keypoints, gray.cols, gray.rows, '#00ff00', false)
  keypoints.delete()
  gray.delete()

  ctx.emit('count', dicts.length)
  return { keypoints: dicts }
}

export const featHarris: NodeImpl = (inputs, params, ctx) => {
  const img = (inputs.image ?? inputs.main) as any
  if (!img) return { main: null, mask: null, keypoints: [], count: 0, points: [] }
  const cv = ctx.cv

  const overlay = ctx.track(toBgr(cv, img))
  const gray = toGray(cv, img)
  const gray32 = new cv.Mat()
  gray.convertTo(gray32, cv.CV_32F)

  const blockSize = Math.max(2, Math.round(Number(params.block_size) || 2))
  let ksize = Math.max(1, Math.round(Number(params.ksize) || 3))
  if (ksize % 2 === 0) ksize += 1
  const k = Number(params.k) ?? 0.04
  const threshold = Number(params.threshold) ?? 0.01

  const response = new cv.Mat()
  cv.cornerHarris(gray32, response, blockSize, ksize, k)

  // Dilating gives, at every pixel, the maximum response in its 3×3 window: the
  // comparison `response === dilated` is then a cheap non-maximum suppression.
  const dilated = new cv.Mat()
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3))
  cv.dilate(response, dilated, kernel)
  kernel.delete()

  const resp = response.data32F
  const dil = dilated.data32F
  let maxResponse = 0
  for (let i = 0; i < dil.length; i++) if (dil[i] > maxResponse) maxResponse = dil[i]
  const cut = maxResponse > 0 ? threshold * maxResponse : 0

  const w = gray.cols
  const h = gray.rows
  const mask = ctx.track(cv.Mat.zeros(h, w, cv.CV_8U))
  const maskData = mask.data
  const overlayData = overlay.data
  const keypoints: KeypointDict[] = []
  const points: { x: number; y: number }[] = []

  for (let i = 0; i < dil.length; i++) {
    if (dil[i] <= cut) continue
    maskData[i] = 255
    overlayData[i * 3] = 0
    overlayData[i * 3 + 1] = 0
    overlayData[i * 3 + 2] = 255
    if (resp[i] !== dil[i]) continue
    const nx = (i % w) / w
    const ny = Math.floor(i / w) / h
    keypoints.push({ _type: 'graphics', shape: 'point', pts: [[nx, ny]], relative: true, color: '#ff3333' })
    points.push({ x: nx, y: ny })
  }

  response.delete()
  dilated.delete()
  gray32.delete()
  gray.delete()

  ctx.emit('count', keypoints.length)
  return { main: overlay, mask, keypoints, count: keypoints.length, points }
}

/* ------------------------------------------------------------------ matching */

/** Side-by-side canvas holding both images, the base for the match drawing. */
function sideBySide(cv: any, left: any, right: any): { canvas: any; offset: number } {
  const a = toBgr(cv, left)
  const b = toBgr(cv, right)
  const canvas = new cv.Mat(Math.max(a.rows, b.rows), a.cols + b.cols, cv.CV_8UC3, new cv.Scalar(0, 0, 0, 255))
  a.copyTo(canvas.roi(new cv.Rect(0, 0, a.cols, a.rows)))
  b.copyTo(canvas.roi(new cv.Rect(a.cols, 0, b.cols, b.rows)))
  const offset = a.cols
  a.delete()
  b.delete()
  return { canvas, offset }
}

/** Pixel coordinates back out of the normalised keypoint dicts. */
function toPixels(dicts: any[], width: number, height: number): [number, number][] {
  return (dicts ?? []).map((d) => {
    const p = d?.pts?.[0] ?? [0, 0]
    return [p[0] * width, p[1] * height] as [number, number]
  })
}

export const featMatcher: NodeImpl = (inputs, params, ctx) => {
  const cv = ctx.cv
  const des1 = inputs.des1 as any
  const des2 = inputs.des2 as any
  const img1 = inputs.img1 as any
  const img2 = inputs.img2 as any

  if (!img1 && !img2) return { main: null, matches_count: 0 }
  if (!des1 || !des2 || !img1 || !img2) return { main: img1 ?? img2, matches_count: 0 }
  if (des1.rows < 2 || des2.rows < 2) {
    const { canvas } = sideBySide(cv, img1, img2)
    return { main: ctx.track(canvas), matches_count: 0 }
  }

  // FLANN is not in this build; brute force is exact anyway, just slower, and at
  // these descriptor counts the difference is not perceptible.
  const normIndex = Math.round(Number(params.norm) ?? 1)
  const norm = normIndex === 0 ? cv.NORM_L2 : cv.NORM_HAMMING
  const ratio = Number(params.ratio_test) || 0.75
  const maxDisplay = Math.max(1, Math.round(Number(params.max_matches) || 50))

  const matcher = new cv.BFMatcher(norm, false)
  const knn = new cv.DMatchVectorVector()
  const good: { queryIdx: number; trainIdx: number; distance: number }[] = []
  try {
    matcher.knnMatch(des1, des2, knn, 2)
    for (let i = 0; i < knn.size(); i++) {
      const pair = knn.get(i)
      if (pair.size() === 2) {
        const best = pair.get(0)
        const second = pair.get(1)
        // Lowe's ratio test: keep a match only when the best is clearly better
        // than the runner-up, which is what rejects ambiguous repeated texture.
        if (best.distance < ratio * second.distance) {
          good.push({ queryIdx: best.queryIdx, trainIdx: best.trainIdx, distance: best.distance })
        }
      }
      pair.delete()
    }
  } catch {
    // Mismatched descriptor types (an L2 norm against binary descriptors, say).
    knn.delete()
    matcher.delete()
    const { canvas } = sideBySide(cv, img1, img2)
    return { main: ctx.track(canvas), matches_count: 0 }
  }
  knn.delete()
  matcher.delete()

  const { canvas, offset } = sideBySide(cv, img1, img2)
  const out = ctx.track(canvas)

  const kp1 = toPixels(inputs.kp1 as any[], img1.cols, img1.rows)
  const kp2 = toPixels(inputs.kp2 as any[], img2.cols, img2.rows)
  if (kp1.length && kp2.length) {
    // cv.KeyPoint is not constructible in this build, so drawMatches is out of
    // reach and the lines are drawn here instead.
    const shown = [...good].sort((a, b) => a.distance - b.distance).slice(0, maxDisplay)
    for (const match of shown) {
      const from = kp1[match.queryIdx]
      const to = kp2[match.trainIdx]
      if (!from || !to) continue
      const colour = new cv.Scalar(((match.trainIdx * 53) % 200) + 55, ((match.queryIdx * 97) % 200) + 55, 200, 255)
      const p1 = new cv.Point(Math.round(from[0]), Math.round(from[1]))
      const p2 = new cv.Point(Math.round(to[0]) + offset, Math.round(to[1]))
      cv.circle(out, p1, 3, colour, 1, cv.LINE_AA)
      cv.circle(out, p2, 3, colour, 1, cv.LINE_AA)
      cv.line(out, p1, p2, colour, 1, cv.LINE_AA)
    }
  }

  ctx.emit('matches_count', good.length)
  return { main: out, matches_count: good.length }
}
