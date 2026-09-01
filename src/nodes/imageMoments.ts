import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'

function huLog(hu: number[]): number[] {
  return hu.map((v) => (v === 0 ? 0 : -Math.sign(v) * Math.log10(Math.abs(v))))
}

export const imageMomentsNode: NodeDef = {
  typeId: 'image_moments',
  label: 'Image Moments',
  category: 'Measure',
  description:
    "Moments spatiaux et invariants de Hu d'une forme : centroïde, orientation θ, anisotropie, ellipse équivalente.",
  inputs: [
    { id: 'image', label: 'image', color: 'image' },
    { id: 'mask', label: 'mask', color: 'mask' },
  ],
  outputs: [
    { id: 'main', label: 'overlay', color: 'image' },
    { id: 'data', label: 'moments', color: 'dict' },
  ],
  params: [
    { id: 'draw_overlay', label: 'Draw Overlay', type: 'boolean', default: true },
    { id: 'draw_ellipse', label: 'Draw Principal Axis', type: 'boolean', default: false },
  ],
  process(inputs, params, ctx) {
    const image = inputs.image as any
    const mask = inputs.mask as any
    if (!image) return { main: undefined, data: undefined }
    const cv = ctx.cv

    const binary = trackMat(new cv.Mat())
    if (mask) {
      const g = trackMat(new cv.Mat())
      if (mask.channels() === 1) mask.copyTo(g)
      else cv.cvtColor(mask, g, cv.COLOR_BGR2GRAY)
      cv.threshold(g, binary, 127, 255, cv.THRESH_BINARY)
    } else {
      const gray = trackMat(new cv.Mat())
      if (image.channels() === 1) image.copyTo(gray)
      else cv.cvtColor(image, gray, cv.COLOR_BGR2GRAY)
      cv.threshold(gray, binary, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    }

    const contours = new cv.MatVector()
    const hierarchy = trackMat(new cv.Mat())
    cv.findContours(binary, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    let moments: any
    let contour: any = null
    if (contours.size() > 0) {
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
      contour = best
      moments = cv.moments(best, false)
    } else {
      moments = cv.moments(binary, true)
    }
    contours.delete()

    const m00 = moments.m00
    const cx = m00 !== 0 ? moments.m10 / m00 : 0
    const cy = m00 !== 0 ? moments.m01 / m00 : 0
    const { mu20, mu02, mu11, mu30, mu03 } = moments

    const eta20 = m00 ? mu20 / m00 ** 2 : 0
    const eta02 = m00 ? mu02 / m00 ** 2 : 0
    const eta11 = m00 ? mu11 / m00 ** 2 : 0

    const theta = 0.5 * ((Math.atan2(2 * mu11, mu20 - mu02) * 180) / Math.PI)
    const denom = mu20 + mu02
    const aniso = denom > 0 ? Math.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2) / denom : 0

    let semiMajor = 0
    let semiMinor = 0
    let eccentricity = 0
    if (mu20 + mu02 > 0) {
      const term = Math.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2)
      const lam1 = (mu20 + mu02 + term) / 2
      const lam2 = (mu20 + mu02 - term) / 2
      semiMajor = 2 * Math.sqrt(Math.max(lam1, 0))
      semiMinor = 2 * Math.sqrt(Math.max(lam2, 0))
      eccentricity = lam1 > 0 && lam2 >= 0 ? Math.sqrt(1 - lam2 / lam1) : 0
    }

    let huArr: number[] = []
    if (cv.HuMoments) {
      const huRaw = cv.HuMoments(moments)
      if (Array.isArray(huRaw)) huArr = huRaw
      else if (huRaw?.data64F) {
        huArr = Array.from(huRaw.data64F as Float64Array)
        huRaw.delete?.()
      }
    }
    const hu = huLog(huArr)

    const overlay = trackMat(new cv.Mat())
    if (image.channels() === 1) cv.cvtColor(image, overlay, cv.COLOR_GRAY2BGR)
    else image.copyTo(overlay)

    if (params.draw_overlay && m00 !== 0) {
      const cxi = Math.round(cx)
      const cyi = Math.round(cy)
      if (contour) {
        const vec = new cv.MatVector()
        vec.push_back(contour)
        cv.drawContours(overlay, vec, -1, new cv.Scalar(0, 255, 255, 255), 2)
        vec.delete()
      }
      cv.circle(overlay, new cv.Point(cxi, cyi), 6, new cv.Scalar(0, 0, 255, 255), -1)
      cv.line(overlay, new cv.Point(cxi - 14, cyi), new cv.Point(cxi + 14, cyi), new cv.Scalar(0, 0, 255, 255), 2)
      cv.line(overlay, new cv.Point(cxi, cyi - 14), new cv.Point(cxi, cyi + 14), new cv.Scalar(0, 0, 255, 255), 2)
    }

    if (params.draw_ellipse && m00 !== 0 && semiMajor > 1) {
      const cxi = Math.round(cx)
      const cyi = Math.round(cy)
      const rad = (theta * Math.PI) / 180
      const dx = Math.round(semiMajor * Math.cos(rad))
      const dy = Math.round(semiMajor * Math.sin(rad))
      cv.ellipse(
        overlay,
        new cv.Point(cxi, cyi),
        new cv.Size(Math.max(1, Math.round(semiMajor)), Math.max(1, Math.round(semiMinor))),
        -theta,
        0,
        360,
        new cv.Scalar(255, 180, 0, 255),
        1,
        cv.LINE_AA
      )
      cv.line(overlay, new cv.Point(cxi - dx, cyi - dy), new cv.Point(cxi + dx, cyi + dy), new cv.Scalar(255, 180, 0, 255), 2)
    }

    return {
      main: overlay,
      data: {
        M00: m00,
        centroid_x: cx,
        centroid_y: cy,
        area: m00,
        mu20,
        mu02,
        mu11,
        mu30,
        mu03,
        eta20,
        eta02,
        eta11,
        theta_deg: theta,
        anisotropy: aniso,
        semi_major: semiMajor,
        semi_minor: semiMinor,
        eccentricity,
        phi1: hu[0] ?? 0,
        phi2: hu[1] ?? 0,
        phi3: hu[2] ?? 0,
        phi4: hu[3] ?? 0,
        phi5: hu[4] ?? 0,
        phi6: hu[5] ?? 0,
        phi7: hu[6] ?? 0,
      },
    }
  },
}
