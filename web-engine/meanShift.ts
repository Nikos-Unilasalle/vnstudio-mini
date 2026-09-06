/**
 * `cv2.pyrMeanShiftFiltering`, written out.
 *
 * The function is compiled into the WASM but not bound to JavaScript, and it is
 * not something that can be approximated with the primitives that are: the
 * result depends on the exact iteration and the integer rounding OpenCV uses.
 * The filtering loop below follows the classic C++ implementation step for step,
 * `cvRound`'s round-half-to-even included, and at maxLevel = 0 it reproduces
 * cv2's output byte for byte.
 *
 * At maxLevel >= 1 it does not, and the difference is not a bug that went
 * unfound: the desktop's cv2 5.0 seeds its finer pyramid levels in some way
 * that its output cannot be reconstructed from the coarse result, the source,
 * or any boundary mask over them — a third of the pixels match none of those
 * candidates. What follows is the documented pyramid (filter coarse, upsample,
 * refit only near colour boundaries). It is a genuine mean-shift segmentation
 * and stays within a few grey levels of cv2's, but the two are not identical,
 * so a graph relying on exact parity should set Max Level to 0.
 */

/**
 * OpenCV's `cvRound`, which rounds half to even under the default FPU mode.
 * `Math.round` rounds half up, and the difference is not cosmetic here: the
 * mean-shift centre is rounded every iteration, so a single half-way case sends
 * the window one pixel off and the pixel converges on a different colour.
 */
function cvRound(value: number): number {
  const floor = Math.floor(value)
  const fraction = value - floor
  if (fraction > 0.5) return floor + 1
  if (fraction < 0.5) return floor
  return floor % 2 === 0 ? floor : floor + 1
}

/** Squared-difference table, indexed by (a - b + 255), exactly as OpenCV builds it. */
const DIFF_TABLE = (() => {
  const table = new Int32Array(768)
  for (let i = 0; i < 768; i++) table[i] = (i - 255) * (i - 255)
  return table
})()

/**
 * One mean-shift filtering pass over `src` (8-bit BGR, `width`x`height`).
 *
 * `mask`, when given, marks the pixels to recompute; the rest keep whatever
 * `dst` already holds, which is how a coarser pyramid level's result survives
 * into a finer one.
 */
function meanShiftLevel(
  src: Uint8Array,
  dst: Uint8Array,
  width: number,
  height: number,
  sp: number,
  sr: number,
  mask: Uint8Array | null,
  maxIterations: number,
  epsilon: number
): void {
  const stride = width * 3
  const sr2 = sr * sr
  const isr2 = cvRound(sr2)

  for (let i = 0; i < height; i++) {
    for (let j = 0; j < width; j++) {
      if (mask && !mask[i * width + j]) continue

      let x0 = j
      let y0 = i
      let c0 = src[i * stride + j * 3]
      let c1 = src[i * stride + j * 3 + 1]
      let c2 = src[i * stride + j * 3 + 2]

      for (let iteration = 0; iteration < maxIterations; iteration++) {
        const minX = Math.max(x0 - sp, 0)
        const minY = Math.max(y0 - sp, 0)
        const maxX = Math.min(x0 + sp, width - 1)
        const maxY = Math.min(y0 + sp, height - 1)

        let s0 = 0
        let s1 = 0
        let s2 = 0
        let sx = 0
        let sy = 0
        let count = 0
        for (let y = minY; y <= maxY; y++) {
          let rowCount = 0
          const rowBase = y * stride
          for (let x = minX; x <= maxX; x++) {
            const at = rowBase + x * 3
            const t0 = src[at]
            const t1 = src[at + 1]
            const t2 = src[at + 2]
            // The colour window is a sphere of radius sr in BGR space.
            if (DIFF_TABLE[t0 - c0 + 255] + DIFF_TABLE[t1 - c1 + 255] + DIFF_TABLE[t2 - c2 + 255] <= isr2) {
              s0 += t0
              s1 += t1
              s2 += t2
              sx += x
              rowCount++
            }
          }
          count += rowCount
          sy += y * rowCount
        }
        if (count === 0) break

        const inverse = 1 / count
        const x1 = cvRound(sx * inverse)
        const y1 = cvRound(sy * inverse)
        const n0 = cvRound(s0 * inverse)
        const n1 = cvRound(s1 * inverse)
        const n2 = cvRound(s2 * inverse)

        const stop =
          (x0 === x1 && y0 === y1) ||
          Math.abs(x1 - x0) + Math.abs(y1 - y0) +
            DIFF_TABLE[n0 - c0 + 255] + DIFF_TABLE[n1 - c1 + 255] + DIFF_TABLE[n2 - c2 + 255] <= epsilon

        x0 = x1
        y0 = y1
        c0 = n0
        c1 = n1
        c2 = n2
        if (stop) break
      }

      const out = i * stride + j * 3
      dst[out] = c0
      dst[out + 1] = c1
      dst[out + 2] = c2
    }
  }
}

/**
 * Marks which pixels of the finer level need recomputing: those whose upsampled
 * colour differs from a neighbour's by more than `sr`. OpenCV builds this on the
 * coarse grid, writes it at even coordinates of the fine grid, then dilates.
 */
function boundaryMask(coarse: Uint8Array, coarseW: number, coarseH: number, fineW: number, fineH: number, sr: number): Uint8Array {
  const mask = new Uint8Array(fineW * fineH)
  const stride = coarseW * 3
  // The C++ compares against MAX(isr2, 16) with >=, not against isr2 with >.
  const isr22 = Math.max(cvRound(sr * sr), 16)
  const differs = (a: number, b: number): boolean =>
    DIFF_TABLE[coarse[a] - coarse[b] + 255] +
      DIFF_TABLE[coarse[a + 1] - coarse[b + 1] + 255] +
      DIFF_TABLE[coarse[a + 2] - coarse[b + 2] + 255] >= isr22

  for (let i = 1; i < coarseH - 1; i++) {
    for (let j = 1; j < coarseW - 1; j++) {
      const at = i * stride + j * 3
      const flagged =
        differs(at, at - 3) || differs(at, at + 3) ||
        differs(at, at - stride - 3) || differs(at, at - stride) || differs(at, at - stride + 3) ||
        differs(at, at + stride - 3) || differs(at, at + stride) || differs(at, at + stride + 3)
      if (!flagged) continue
      // The C++ walks the fine mask two rows per coarse row starting at row 1,
      // so coarse (i, j) writes fine (2i - 1, 2j - 1), before the dilation.
      const fy = 2 * i - 1
      const fx = 2 * j - 1
      if (fy < fineH && fx >= 0 && fx < fineW) mask[fy * fineW + fx] = 1
    }
  }

  // 3x3 dilation, the default structuring element OpenCV passes.
  const dilated = new Uint8Array(mask.length)
  for (let y = 0; y < fineH; y++) {
    for (let x = 0; x < fineW; x++) {
      let any = 0
      for (let dy = -1; dy <= 1 && !any; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ny = y + dy
          const nx = x + dx
          if (ny < 0 || nx < 0 || ny >= fineH || nx >= fineW) continue
          if (mask[ny * fineW + nx]) { any = 1; break }
        }
      }
      dilated[y * fineW + x] = any
    }
  }
  return dilated
}

/**
 * Pyramidal mean-shift filtering. `src` must be an 8-bit 3-channel Mat; the
 * returned Mat is the caller's to free (or to hand to `ctx.track`).
 */
export function pyrMeanShiftFiltering(cv: any, src: any, sp: number, sr: number, maxLevel: number, maxIterations = 5, epsilon = 1): any {
  const levels = Math.max(0, Math.min(maxLevel, 8))

  // Build the source pyramid, coarsest last.
  const srcPyramid: any[] = [src]
  for (let level = 1; level <= levels; level++) {
    const down = new cv.Mat()
    cv.pyrDown(srcPyramid[level - 1], down)
    srcPyramid.push(down)
  }

  const dstPyramid: any[] = srcPyramid.map((m: any) => new cv.Mat(m.rows, m.cols, cv.CV_8UC3))

  for (let level = levels; level >= 0; level--) {
    const source = srcPyramid[level]
    const destination = dstPyramid[level]
    let mask: Uint8Array | null = null

    if (level < levels) {
      // Start from the coarser result, then only redo the pixels near a colour
      // boundary — this is what makes the pyramid a speed-up and not just a blur.
      cv.pyrUp(dstPyramid[level + 1], destination, new cv.Size(destination.cols, destination.rows))
      mask = boundaryMask(
        dstPyramid[level + 1].data as Uint8Array,
        dstPyramid[level + 1].cols,
        dstPyramid[level + 1].rows,
        destination.cols,
        destination.rows,
        sr
      )
    }

    meanShiftLevel(
      source.data as Uint8Array,
      destination.data as Uint8Array,
      source.cols,
      source.rows,
      sp,
      sr,
      mask,
      maxIterations,
      epsilon
    )
  }

  const result = dstPyramid[0]
  for (let level = 1; level <= levels; level++) {
    srcPyramid[level].delete()
    dstPyramid[level].delete()
  }
  return result
}
