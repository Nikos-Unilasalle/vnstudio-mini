import type { NodeDef } from '../engine/types'
import { trackMat } from '../engine/executor'

const ZONES_4: [string, [number, number, number]][] = [
  ['Toes', [80, 200, 120]],
  ['Forefoot', [80, 160, 240]],
  ['Arch', [240, 160, 60]],
  ['Heel', [240, 80, 60]],
]
const ZONES_3: [string, [number, number, number]][] = [
  ['Forefoot', [80, 200, 120]],
  ['Arch', [80, 160, 240]],
  ['Heel', [240, 80, 60]],
]

function extents(binary: any): { ymi: number; yma: number; xmi: number; xma: number } {
  const data = binary.data as Uint8Array
  const w = binary.cols
  const h = binary.rows
  let ymi = h
  let yma = -1
  let xmi = w
  let xma = -1
  for (let y = 0; y < h; y++) {
    const row = y * w
    for (let x = 0; x < w; x++) {
      if (data[row + x] > 0) {
        if (y < ymi) ymi = y
        if (y > yma) yma = y
        if (x < xmi) xmi = x
        if (x > xma) xma = x
      }
    }
  }
  if (yma < 0) return { ymi: 0, yma: 0, xmi: 0, xma: 0 }
  return { ymi, yma, xmi, xma }
}

function standUpright(cv: any, vis: any, gray: any, binary: any): { vis: any; gray: any; binary: any } {
  const rot = (v: any, g: any, b: any, code: number) => {
    const v2 = new cv.Mat()
    const g2 = new cv.Mat()
    const b2 = new cv.Mat()
    cv.rotate(v, v2, code)
    cv.rotate(g, g2, code)
    cv.rotate(b, b2, code)
    return [v2, g2, b2]
  }

  let e = extents(binary)
  if (e.xma - e.xmi > e.yma - e.ymi) {
    ;[vis, gray, binary] = rot(vis, gray, binary, cv.ROTATE_90_CLOCKWISE)
    e = extents(binary)
  }

  const w = binary.cols
  const data = binary.data as Uint8Array
  let maxCount = -1
  let maxRow = 0
  for (let y = e.ymi; y <= e.yma; y++) {
    let count = 0
    const row = y * w
    for (let x = 0; x < w; x++) if (data[row + x] > 0) count++
    if (count > maxCount) {
      maxCount = count
      maxRow = y - e.ymi
    }
  }
  const bandHeight = e.yma - e.ymi + 1
  if (maxCount > 0 && maxRow > bandHeight / 2) {
    ;[vis, gray, binary] = rot(vis, gray, binary, cv.ROTATE_180)
  }
  return { vis, gray, binary }
}

export const forensicFootprintNode: NodeDef = {
  typeId: 'forensic_footprint',
  label: 'Barefoot Print Forensics',
  category: 'Analysis',
  description:
    "Analyse forensique d'une empreinte de pied NU sur un recadrage redressé : zones de pression, TFL/CBW/HBW en mm, indice de Staheli, asymétrie. Branche les sorties rotated/rotated_mask d'un Oriented Bounding Box, et px_per_mm d'une calibration.",
  inputs: [
    { id: 'image', label: 'image', color: 'image' },
    { id: 'mask', label: 'mask', color: 'mask' },
    { id: 'px_per_mm', label: 'Px/mm', color: 'scalar' },
  ],
  outputs: [
    { id: 'main', label: 'overlay', color: 'image' },
    { id: 'report', label: 'report', color: 'dict' },
    { id: 'staheli', label: 'staheli', color: 'scalar' },
    { id: 'asymmetry', label: 'asymmetry', color: 'scalar' },
  ],
  params: [
    {
      id: 'n_zones',
      label: 'Zones',
      type: 'select',
      default: 0,
      options: [
        { label: '4 zones (Toes/FF/Arch/Heel)', value: 0 },
        { label: '3 zones (FF/Arch/Heel)', value: 1 },
      ],
    },
    { id: 'pressure_weights', label: 'Pressure Weights', type: 'boolean', default: true },
    { id: 'show_measurements', label: 'Width Lines', type: 'boolean', default: true },
    { id: 'alpha', label: 'Overlay Alpha', type: 'number', default: 0.55, min: 0, max: 1, step: 0.05 },
  ],
  process(inputs, params, ctx) {
    const image = inputs.image as any
    const maskIn = inputs.mask as any
    if (!image) return {}
    const cv = ctx.cv
    const pxPerMmRaw = inputs.px_per_mm as number | undefined
    const pxPerMm = typeof pxPerMmRaw === 'number' && pxPerMmRaw > 0 ? pxPerMmRaw : 0
    const hasCalib = pxPerMm > 0

    const zoneDefs = Number(params.n_zones) === 0 ? ZONES_4 : ZONES_3
    const n = zoneDefs.length

    let visSrc = trackMat(new cv.Mat())
    if (image.channels() === 1) cv.cvtColor(image, visSrc, cv.COLOR_GRAY2BGR)
    else image.copyTo(visSrc)

    let grayImg = trackMat(new cv.Mat())
    cv.cvtColor(visSrc, grayImg, cv.COLOR_BGR2GRAY)

    let binary = trackMat(new cv.Mat())
    if (maskIn) {
      const mg = trackMat(new cv.Mat())
      if (maskIn.channels() === 1) maskIn.copyTo(mg)
      else cv.cvtColor(maskIn, mg, cv.COLOR_BGR2GRAY)
      cv.threshold(mg, binary, 127, 255, cv.THRESH_BINARY)
    } else {
      cv.threshold(grayImg, binary, 1, 255, cv.THRESH_BINARY)
    }

    let e0 = extents(binary)
    if (e0.xma < 0) return { main: visSrc, report: {}, staheli: 0, asymmetry: 0 }

    const upright = standUpright(cv, visSrc, grayImg, binary)
    visSrc = trackMat(upright.vis)
    grayImg = trackMat(upright.gray)
    binary = trackMat(upright.binary)

    const { ymi, yma, xmi, xma } = extents(binary)
    const fh = Math.max(1, yma - ymi)
    const fw = Math.max(1, xma - xmi)

    const bounds: number[] = []
    for (let i = 0; i <= n; i++) bounds.push(ymi + Math.floor((i * fh) / n))
    bounds[n] = yma

    const ov = trackMat(visSrc.clone())
    const w = binary.cols
    const bdata = binary.data as Uint8Array
    let totalArea = 0
    for (let i = 0; i < bdata.length; i++) if (bdata[i] > 0) totalArea++
    totalArea = Math.max(1, totalArea)

    const metrics: Record<string, number | string> = {}
    const zoneAreas: number[] = []

    for (let i = 0; i < n; i++) {
      const [name, col] = zoneDefs[i]
      const y0 = bounds[i]
      const y1 = bounds[i + 1]
      let area = 0
      for (let y = y0; y < y1; y++) {
        const row = y * w
        for (let x = xmi; x < xma; x++) {
          if (bdata[row + x] > 0) {
            area++
            const off = row * 3 + x * 3
            ;(ov.data as Uint8Array)[off] = col[2] // B
            ;(ov.data as Uint8Array)[off + 1] = col[1] // G
            ;(ov.data as Uint8Array)[off + 2] = col[0] // R
          }
        }
      }
      zoneAreas.push(area)
      const pct = Math.round((1000 * area) / totalArea) / 10
      metrics[name.toLowerCase() + '_area_pct'] = pct

      const fs = Math.max(0.3, (0.55 * fw) / 200)
      cv.putText(ov, `${name}: ${pct}%`, new cv.Point(xmi + 6, Math.floor((y0 + y1) / 2)), cv.FONT_HERSHEY_SIMPLEX, fs, new cv.Scalar(255, 255, 255, 255), Math.max(1, Math.floor(fw / 150)), cv.LINE_AA)
      if (i < n - 1) cv.line(ov, new cv.Point(xmi, y1), new cv.Point(xma, y1), new cv.Scalar(200, 200, 200, 255), 1)
    }

    let archA: number
    let nonHeel: number
    if (n === 4) {
      archA = zoneAreas[2]
      nonHeel = Math.max(1, zoneAreas[1] + zoneAreas[2])
    } else {
      archA = zoneAreas[1]
      nonHeel = Math.max(1, zoneAreas[0] + zoneAreas[1])
    }
    const staheli = Math.round((archA / nonHeel) * 1000) / 1000
    const archType = staheli < 0.21 ? 'Cavus' : staheli < 0.26 ? 'Normal' : 'Flat'
    metrics.staheli_arch_index = staheli
    metrics.arch_type = archType

    const midX = Math.floor((xmi + xma) / 2)
    let la = 0
    let ra = 0
    for (let y = 0; y < binary.rows; y++) {
      const row = y * w
      for (let x = xmi; x < midX; x++) if (bdata[row + x] > 0) la++
      for (let x = midX; x < xma; x++) if (bdata[row + x] > 0) ra++
    }
    const asym = Math.round((Math.abs(la - ra) / Math.max(la + ra, 1)) * 1000) / 1000
    metrics.asymmetry_score = asym

    let cxc: number
    let cyc: number
    if (params.pressure_weights) {
      const gdata = grayImg.data as Uint8Array
      let totalW = 0
      let sumX = 0
      let sumY = 0
      for (let y = 0; y < binary.rows; y++) {
        const row = y * w
        for (let x = 0; x < w; x++) {
          if (bdata[row + x] > 0) {
            const wt = gdata[row + x]
            totalW += wt
            sumX += x * wt
            sumY += y * wt
          }
        }
      }
      if (totalW > 0) {
        cxc = Math.round(sumX / totalW)
        cyc = Math.round(sumY / totalW)
      } else {
        cxc = Math.round((xmi + xma) / 2)
        cyc = Math.round((ymi + yma) / 2)
      }
    } else {
      cxc = Math.round((xmi + xma) / 2)
      cyc = Math.round((ymi + yma) / 2)
    }

    metrics.centroid_x_pct = Math.round((1000 * (cxc - xmi)) / fw) / 10
    metrics.centroid_y_pct = Math.round((1000 * (cyc - ymi)) / fh) / 10
    metrics.total_area_px = totalArea
    if (hasCalib) {
      metrics.foot_length_mm = Math.round((fh / pxPerMm) * 10) / 10
      metrics.foot_width_mm = Math.round((fw / pxPerMm) * 10) / 10
    }

    const zoneNames = zoneDefs.map((z) => z[0])
    const widths: Record<string, number> = {}
    for (const [key, zoneName] of [
      ['forefoot', 'Forefoot'],
      ['heel', 'Heel'],
    ] as const) {
      const zi = zoneNames.indexOf(zoneName)
      if (zi < 0) continue
      const y0 = bounds[zi]
      const y1 = bounds[zi + 1]
      let bestRow = -1
      let bestCount = -1
      for (let y = y0; y < y1; y++) {
        const row = y * w
        let count = 0
        for (let x = xmi; x < xma; x++) if (bdata[row + x] > 0) count++
        if (count > bestCount) {
          bestCount = count
          bestRow = y
        }
      }
      if (bestCount <= 0) continue
      const row = bestRow * w
      let lx = -1
      let rx = -1
      for (let x = xmi; x < xma; x++) {
        if (bdata[row + x] > 0) {
          if (lx < 0) lx = x
          rx = x
        }
      }
      const wPx = rx - lx
      widths[key] = wPx
      metrics[key + '_width_px'] = wPx
      if (hasCalib) metrics[key + '_width_mm'] = Math.round((wPx / pxPerMm) * 10) / 10
      if (params.show_measurements) {
        cv.line(ov, new cv.Point(lx, bestRow), new cv.Point(rx, bestRow), new cv.Scalar(0, 165, 255, 255), Math.max(1, Math.floor(fw / 200)))
        const fs2 = Math.max(0.25, (0.38 * fw) / 200)
        const label = hasCalib ? `${(wPx / pxPerMm).toFixed(1)}mm` : `${wPx}px`
        cv.putText(ov, label, new cv.Point(rx + 4, bestRow), cv.FONT_HERSHEY_SIMPLEX, fs2, new cv.Scalar(0, 165, 255, 255), 1, cv.LINE_AA)
      }
    }
    if (widths.forefoot > 0) metrics.heel_forefoot_ratio = Math.round(((widths.heel ?? 0) / widths.forefoot) * 1000) / 1000

    cv.line(ov, new cv.Point(midX, ymi), new cv.Point(midX, yma), new cv.Scalar(220, 220, 0, 255), Math.max(1, Math.floor(fw / 120)))

    const rd = Math.max(5, Math.floor(fw / 50))
    cv.circle(ov, new cv.Point(cxc, cyc), rd, new cv.Scalar(0, 255, 255, 255), -1)
    cv.circle(ov, new cv.Point(cxc, cyc), rd, new cv.Scalar(0, 0, 0, 255), 2)

    const fs3 = Math.max(0.3, (0.45 * fw) / 200)
    const th3 = Math.max(1, Math.floor(fw / 180))
    const lineH = Math.round((18 * fw) / 200)
    cv.putText(ov, `Staheli ${staheli} - ${archType}`, new cv.Point(xmi + 5, yma - 8), cv.FONT_HERSHEY_SIMPLEX, fs3, new cv.Scalar(100, 255, 255, 255), th3, cv.LINE_AA)
    cv.putText(ov, `Asym ${asym}`, new cv.Point(xmi + 5, yma - 8 - lineH), cv.FONT_HERSHEY_SIMPLEX, fs3, new cv.Scalar(220, 220, 0, 255), th3, cv.LINE_AA)

    const alpha = Number(params.alpha)
    const final = trackMat(new cv.Mat())
    cv.addWeighted(ov, alpha, visSrc, 1 - alpha, 0, final)

    return { main: final, report: metrics, staheli, asymmetry: asym }
  },
}
