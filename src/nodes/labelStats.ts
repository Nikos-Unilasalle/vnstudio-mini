export interface LabelStat {
  id: number
  area: number
  cx: number
  cy: number
  minX: number
  minY: number
  maxX: number
  maxY: number
}

/** Scans a CV_32S label matrix once and returns per-label stats. Labels <= 0 (background / boundary) are skipped. */
export function computeLabelStats(labelsMat: any): Map<number, LabelStat> {
  const data = labelsMat.data32S as Int32Array
  const w = labelsMat.cols
  const h = labelsMat.rows
  const acc = new Map<number, { area: number; sumX: number; sumY: number; minX: number; minY: number; maxX: number; maxY: number }>()

  for (let y = 0; y < h; y++) {
    const row = y * w
    for (let x = 0; x < w; x++) {
      const label = data[row + x]
      if (label <= 0) continue
      let s = acc.get(label)
      if (!s) {
        s = { area: 0, sumX: 0, sumY: 0, minX: x, minY: y, maxX: x, maxY: y }
        acc.set(label, s)
      }
      s.area++
      s.sumX += x
      s.sumY += y
      if (x < s.minX) s.minX = x
      if (x > s.maxX) s.maxX = x
      if (y < s.minY) s.minY = y
      if (y > s.maxY) s.maxY = y
    }
  }

  const out = new Map<number, LabelStat>()
  for (const [id, s] of acc) {
    out.set(id, {
      id,
      area: s.area,
      cx: s.sumX / s.area,
      cy: s.sumY / s.area,
      minX: s.minX,
      minY: s.minY,
      maxX: s.maxX,
      maxY: s.maxY,
    })
  }
  return out
}

/** Builds a stable pseudo-random color for a label id, for colorized previews. */
export function labelColor(id: number): [number, number, number] {
  const h = (id * 2654435761) % 360
  const hue = ((h % 360) + 360) % 360
  return hslToRgb(hue, 65, 55)
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  s /= 100
  l /= 100
  const k = (n: number) => (n + h / 30) % 12
  const a = s * Math.min(l, 1 - l)
  const f = (n: number) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)))
  return [Math.round(255 * f(0)), Math.round(255 * f(8)), Math.round(255 * f(4))]
}
