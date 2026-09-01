import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'

export const geomObbNode: NodeDef = {
  typeId: 'geom_obb',
  label: 'Oriented Bounding Box',
  category: 'Geometry',
  description: "Boîte englobante orientée à aire minimale, calculée sur les contours du masque, et recadrage redressé.",
  inputs: [
    { id: 'image', label: 'image', color: 'image' },
    { id: 'mask', label: 'mask', color: 'mask' },
  ],
  outputs: [
    { id: 'main', label: 'image (annotée)', color: 'image' },
    { id: 'rotated', label: 'rotated', color: 'image' },
    { id: 'rotated_mask', label: 'rotated_mask', color: 'mask' },
    { id: 'angle', label: 'angle', color: 'scalar' },
  ],
  params: [
    { id: 'draw_obb', label: 'Draw OBB', type: 'boolean', default: true },
    { id: 'auto_crop', label: 'Crop to OBB', type: 'boolean', default: true },
    {
      id: 'target',
      label: 'Target',
      type: 'select',
      default: 'largest',
      options: [
        { label: 'largest', value: 'largest' },
        { label: 'all', value: 'all' },
        { label: 'combined', value: 'combined' },
      ],
    },
  ],
  process(inputs, params, ctx) {
    const image = inputs.image as any
    const mask = inputs.mask as any
    if (!image) return { main: undefined, rotated: undefined, rotated_mask: undefined, angle: 0 }
    const cv = ctx.cv

    const vis = trackMat(new cv.Mat())
    if (image.channels() === 1) cv.cvtColor(image, vis, cv.COLOR_GRAY2BGR)
    else image.copyTo(vis)
    const W = vis.cols
    const H = vis.rows

    let src: any
    if (mask) {
      src = trackMat(new cv.Mat())
      if (mask.channels() === 1) mask.copyTo(src)
      else cv.cvtColor(mask, src, cv.COLOR_BGR2GRAY)
    } else {
      const gray = trackMat(new cv.Mat())
      cv.cvtColor(vis, gray, cv.COLOR_BGR2GRAY)
      src = trackMat(new cv.Mat())
      cv.threshold(gray, src, 1, 255, cv.THRESH_BINARY)
    }

    const contours = new cv.MatVector()
    const hierarchy = trackMat(new cv.Mat())
    cv.findContours(src, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if (contours.size() === 0) {
      contours.delete()
      return { main: vis, rotated: vis, rotated_mask: src, angle: 0 }
    }

    const target = params.target as string
    let groups: any[]
    if (target === 'largest') {
      let best = contours.get(0)
      let bestArea = cv.contourArea(best)
      for (let i = 1; i < contours.size(); i++) {
        const c = contours.get(i)
        const a = cv.contourArea(c)
        if (a > bestArea) {
          bestArea = a
          best = c
        }
      }
      groups = [best]
    } else if (target === 'combined') {
      const allPts: number[] = []
      for (let i = 0; i < contours.size(); i++) {
        const c = contours.get(i)
        const data = c.data32S as Int32Array
        for (let j = 0; j < data.length; j++) allPts.push(data[j])
      }
      groups = [cv.matFromArray(allPts.length / 2, 1, cv.CV_32SC2, allPts)]
    } else {
      groups = []
      for (let i = 0; i < contours.size(); i++) groups.push(contours.get(i))
    }

    let lastRect: any = null
    if (params.draw_obb) {
      for (const cnt of groups) {
        const rect = cv.minAreaRect(cnt)
        lastRect = rect
        const boxPts = rotatedRectPoints(rect)
        const pts = boxPts.flatMap((p) => [Math.round(p.x), Math.round(p.y)])
        const boxMat = cv.matFromArray(4, 1, cv.CV_32SC2, pts)
        const vec = new cv.MatVector()
        vec.push_back(boxMat)
        cv.polylines(vis, vec, true, new cv.Scalar(0, 255, 136, 255), 2, cv.LINE_AA)
        boxMat.delete()
        vec.delete()
      }
    } else {
      for (const cnt of groups) lastRect = cv.minAreaRect(cnt)
    }
    contours.delete()

    if (!lastRect) return { main: vis, rotated: vis, rotated_mask: src, angle: 0 }

    let { width: wRect, height: hRect } = lastRect.size
    let angle = lastRect.angle
    if (wRect < hRect) {
      angle += 90
      ;[wRect, hRect] = [hRect, wRect]
    }

    let rotatedOut = vis
    let rotatedMaskOut = src
    if (params.auto_crop && wRect > 0 && hRect > 0) {
      const pad = Math.round(Math.max(W, H) * 0.75)
      const cx = lastRect.center.x + pad
      const cy = lastRect.center.y + pad
      rotatedOut = trackMat(warpCrop(cv, vis, cx, cy, angle, wRect, hRect, pad, cv.INTER_LINEAR))
      rotatedMaskOut = trackMat(warpCrop(cv, src, cx, cy, angle, wRect, hRect, pad, cv.INTER_NEAREST))
    }

    return { main: vis, rotated: rotatedOut, rotated_mask: rotatedMaskOut, angle }
  },
}

function rotatedRectPoints(rect: { center: { x: number; y: number }; size: { width: number; height: number }; angle: number }): { x: number; y: number }[] {
  const { center, size, angle } = rect
  const rad = (angle * Math.PI) / 180
  const b = Math.cos(rad) * 0.5
  const a = Math.sin(rad) * 0.5
  const p0 = { x: center.x - a * size.height - b * size.width, y: center.y + b * size.height - a * size.width }
  const p1 = { x: center.x + a * size.height - b * size.width, y: center.y - b * size.height - a * size.width }
  const p2 = { x: 2 * center.x - p0.x, y: 2 * center.y - p0.y }
  const p3 = { x: 2 * center.x - p1.x, y: 2 * center.y - p1.y }
  return [p0, p1, p2, p3]
}

function warpCrop(cv: any, img: any, cx: number, cy: number, angle: number, wRect: number, hRect: number, pad: number, interp: number): any {
  const padded = new cv.Mat()
  cv.copyMakeBorder(img, padded, pad, pad, pad, pad, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0))
  const M = cv.getRotationMatrix2D(new cv.Point(cx, cy), angle, 1.0)
  const warped = new cv.Mat()
  cv.warpAffine(padded, warped, M, new cv.Size(padded.cols, padded.rows), interp, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0))
  padded.delete()
  M.delete()

  const x0 = Math.max(0, Math.round(cx - wRect / 2))
  const y0 = Math.max(0, Math.round(cy - hRect / 2))
  const x1 = Math.min(warped.cols, Math.round(cx + wRect / 2))
  const y1 = Math.min(warped.rows, Math.round(cy + hRect / 2))
  if (x1 > x0 && y1 > y0) {
    const roi = warped.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0))
    const out = roi.clone()
    warped.delete()
    return out
  }
  return warped
}
