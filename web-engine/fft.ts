/**
 * Fourier and cosine transforms.
 *
 * OpenCV's `dft` and `dct` are compiled into the WASM but not bound to
 * JavaScript by this build, so the transforms are written out here. Images come
 * in arbitrary sizes, and the desktop nodes transform the image at its exact
 * dimensions, so a power-of-two-only FFT would not do: lengths that are not a
 * power of two go through Bluestein's algorithm, which turns any DFT into a
 * convolution that a power-of-two FFT can carry out.
 */

/** True when n is a power of two, the case the fast path handles directly. */
function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0
}

/** In-place radix-2 Cooley-Tukey. `re`/`im` must have a power-of-two length. */
function fftRadix2(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length
  if (n <= 1) return

  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t
      t = im[i]; im[i] = im[j]; im[j] = t
    }
  }

  const sign = inverse ? 1 : -1
  for (let len = 2; len <= n; len <<= 1) {
    const angle = (sign * 2 * Math.PI) / len
    const wRe = Math.cos(angle)
    const wIm = Math.sin(angle)
    for (let i = 0; i < n; i += len) {
      let curRe = 1
      let curIm = 0
      for (let k = 0; k < len / 2; k++) {
        const aRe = re[i + k]
        const aIm = im[i + k]
        const bRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm
        const bIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe
        re[i + k] = aRe + bRe
        im[i + k] = aIm + bIm
        re[i + k + len / 2] = aRe - bRe
        im[i + k + len / 2] = aIm - bIm
        const nextRe = curRe * wRe - curIm * wIm
        curIm = curRe * wIm + curIm * wRe
        curRe = nextRe
      }
    }
  }
}

/**
 * Bluestein (chirp-z): rewrites an arbitrary-length DFT as a convolution of
 * two chirp sequences, which a padded power-of-two FFT then evaluates.
 */
function fftBluestein(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length
  let m = 1
  while (m < 2 * n + 1) m <<= 1

  // The chirp tables carry +πi²/n for the forward transform; the a/b/output
  // formulas below already apply the negative exponent the forward DFT needs,
  // so folding the sign into the table as well would produce the conjugate.
  const direction = inverse ? -1 : 1
  const cosTable = new Float64Array(n)
  const sinTable = new Float64Array(n)
  for (let i = 0; i < n; i++) {
    // The exponent is i² mod 2n, kept small so cos/sin stay accurate.
    const j = (i * i) % (n * 2)
    const angle = (direction * Math.PI * j) / n
    cosTable[i] = Math.cos(angle)
    sinTable[i] = Math.sin(angle)
  }

  const aRe = new Float64Array(m)
  const aIm = new Float64Array(m)
  for (let i = 0; i < n; i++) {
    aRe[i] = re[i] * cosTable[i] + im[i] * sinTable[i]
    aIm[i] = -re[i] * sinTable[i] + im[i] * cosTable[i]
  }

  const bRe = new Float64Array(m)
  const bIm = new Float64Array(m)
  bRe[0] = cosTable[0]
  bIm[0] = sinTable[0]
  for (let i = 1; i < n; i++) {
    bRe[i] = bRe[m - i] = cosTable[i]
    bIm[i] = bIm[m - i] = sinTable[i]
  }

  fftRadix2(aRe, aIm, false)
  fftRadix2(bRe, bIm, false)
  for (let i = 0; i < m; i++) {
    const tRe = aRe[i] * bRe[i] - aIm[i] * bIm[i]
    aIm[i] = aRe[i] * bIm[i] + aIm[i] * bRe[i]
    aRe[i] = tRe
  }
  fftRadix2(aRe, aIm, true)
  for (let i = 0; i < m; i++) {
    aRe[i] /= m
    aIm[i] /= m
  }

  for (let i = 0; i < n; i++) {
    re[i] = aRe[i] * cosTable[i] + aIm[i] * sinTable[i]
    im[i] = -aRe[i] * sinTable[i] + aIm[i] * cosTable[i]
  }
}

/** In-place 1-D DFT of any length. Not normalised — the caller divides. */
export function fft1d(re: Float64Array, im: Float64Array, inverse: boolean): void {
  if (re.length <= 1) return
  if (isPowerOfTwo(re.length)) fftRadix2(re, im, inverse)
  else fftBluestein(re, im, inverse)
}

/**
 * In-place 2-D DFT, rows then columns. `inverse` divides by w*h, matching
 * numpy's `ifft2` convention.
 */
export function fft2d(re: Float64Array, im: Float64Array, w: number, h: number, inverse: boolean): void {
  const rowRe = new Float64Array(w)
  const rowIm = new Float64Array(w)
  for (let y = 0; y < h; y++) {
    const at = y * w
    rowRe.set(re.subarray(at, at + w))
    rowIm.set(im.subarray(at, at + w))
    fft1d(rowRe, rowIm, inverse)
    re.set(rowRe, at)
    im.set(rowIm, at)
  }

  const colRe = new Float64Array(h)
  const colIm = new Float64Array(h)
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) {
      colRe[y] = re[y * w + x]
      colIm[y] = im[y * w + x]
    }
    fft1d(colRe, colIm, inverse)
    for (let y = 0; y < h; y++) {
      re[y * w + x] = colRe[y]
      im[y * w + x] = colIm[y]
    }
  }

  if (inverse) {
    const scale = 1 / (w * h)
    for (let i = 0; i < re.length; i++) {
      re[i] *= scale
      im[i] *= scale
    }
  }
}

/** Moves the zero frequency to the centre, as numpy's `fftshift` does. */
export function fftShift(data: Float64Array, w: number, h: number): Float64Array {
  const out = new Float64Array(data.length)
  const cx = Math.floor(w / 2)
  const cy = Math.floor(h / 2)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      out[((y + cy) % h) * w + ((x + cx) % w)] = data[y * w + x]
    }
  }
  return out
}

/** The exact inverse of fftShift, which differs for odd sizes. */
export function ifftShift(data: Float64Array, w: number, h: number): Float64Array {
  const out = new Float64Array(data.length)
  const cx = Math.floor((w + 1) / 2)
  const cy = Math.floor((h + 1) / 2)
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      out[((y + cy) % h) * w + ((x + cx) % w)] = data[y * w + x]
    }
  }
  return out
}

/* ------------------------------------------------------------------- DCT */

/**
 * Orthonormal 1-D DCT-II, the transform OpenCV's `cv.dct` performs.
 *
 * Computed through the even mirror: extending x to length 2N with
 * z[2N-1-n] = z[n] makes its DFT satisfy Z[k] = 2·S[k]·e^{iπk/2N}, where S is
 * the raw cosine sum. One length-2N FFT therefore yields every coefficient,
 * and the same identity read backwards gives the exact inverse.
 */
function dct1d(input: Float64Array): Float64Array {
  const n = input.length
  const re = new Float64Array(2 * n)
  const im = new Float64Array(2 * n)
  for (let i = 0; i < n; i++) {
    re[i] = input[i]
    re[2 * n - 1 - i] = input[i]
  }
  fft1d(re, im, false)

  const out = new Float64Array(n)
  const scale0 = Math.sqrt(1 / n)
  const scale = Math.sqrt(2 / n)
  for (let k = 0; k < n; k++) {
    const angle = (-Math.PI * k) / (2 * n)
    // Re(Z[k]·e^{-iπk/2N}) is 2·S[k]; the orthonormal factor is applied on top.
    const s = (re[k] * Math.cos(angle) - im[k] * Math.sin(angle)) / 2
    out[k] = (k === 0 ? scale0 : scale) * s
  }
  return out
}

/** Orthonormal 1-D DCT-III, the exact inverse of dct1d. */
function idct1d(input: Float64Array): Float64Array {
  const n = input.length
  const re = new Float64Array(2 * n)
  const im = new Float64Array(2 * n)
  const scale0 = Math.sqrt(1 / n)
  const scale = Math.sqrt(2 / n)

  for (let k = 0; k < n; k++) {
    const s = input[k] / (k === 0 ? scale0 : scale)
    const angle = (Math.PI * k) / (2 * n)
    const zRe = 2 * s * Math.cos(angle)
    const zIm = 2 * s * Math.sin(angle)
    re[k] = zRe
    im[k] = zIm
    // z is real, so the upper half of its spectrum is the conjugate mirror.
    if (k > 0) {
      re[2 * n - k] = zRe
      im[2 * n - k] = -zIm
    }
  }
  // fft1d leaves the inverse transform unnormalised, so the 1/(2N) of the
  // length-2N IDFT is applied here.
  fft1d(re, im, true)
  const norm = 1 / (2 * n)

  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = re[i] * norm
  return out
}

/** Separable 2-D orthonormal DCT-II. */
export function dct2d(data: Float64Array, w: number, h: number): Float64Array {
  const out = new Float64Array(data.length)
  const row = new Float64Array(w)
  for (let y = 0; y < h; y++) {
    row.set(data.subarray(y * w, y * w + w))
    out.set(dct1d(row), y * w)
  }
  const col = new Float64Array(h)
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) col[y] = out[y * w + x]
    const transformed = dct1d(col)
    for (let y = 0; y < h; y++) out[y * w + x] = transformed[y]
  }
  return out
}

/** Separable 2-D orthonormal DCT-III, the inverse of dct2d. */
export function idct2d(data: Float64Array, w: number, h: number): Float64Array {
  const out = new Float64Array(data)
  const col = new Float64Array(h)
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) col[y] = out[y * w + x]
    const transformed = idct1d(col)
    for (let y = 0; y < h; y++) out[y * w + x] = transformed[y]
  }
  const row = new Float64Array(w)
  for (let y = 0; y < h; y++) {
    row.set(out.subarray(y * w, y * w + w))
    out.set(idct1d(row), y * w)
  }
  return out
}
