/**
 * The range / null-space split of an MRI undersampling operator, in the browser.
 *
 * This is the project's Stage 2 made visible. For a k-space mask `M`, the forward operator
 * is `A = M F`, and because `M` is a selection in the Fourier domain the two projectors are
 * diagonal there:
 *
 *     P_R x = F^-1 (M · F x)        what the scanner measures
 *     P_N x = F^-1 ((1 - M) · F x)  what it cannot see, at all
 *
 * The point of showing it is the consequence: anything added inside `P_N` leaves `M · F x`
 * **bit-identical**. It is invisible to every data-consistency check, which is exactly why
 * hallucination in a generative reconstruction cannot be caught by comparing against the
 * measurements. `measurementError` below reports that residual so the claim can be checked
 * rather than taken on trust.
 *
 * Radix-2 Cooley-Tukey, applied along rows then columns. Sizes are powers of two because
 * the demo controls the canvas size; this is not a general FFT library.
 */

/** In-place complex FFT of one row. `inverse` divides by n. */
function fft1d(re, im, inverse) {
  const n = re.length
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      ;[re[i], re[j]] = [re[j], re[i]]
      ;[im[i], im[j]] = [im[j], im[i]]
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((inverse ? 2 : -2) * Math.PI) / len
    const wr = Math.cos(ang)
    const wi = Math.sin(ang)
    for (let i = 0; i < n; i += len) {
      let cr = 1
      let ci = 0
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k]
        const ui = im[i + k]
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr
        re[i + k] = ur + vr
        im[i + k] = ui + vi
        re[i + k + len / 2] = ur - vr
        im[i + k + len / 2] = ui - vi
        const nr = cr * wr - ci * wi
        ci = cr * wi + ci * wr
        cr = nr
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n }
}

/** 2D FFT over an n x n image held as flat Float64Arrays. */
export function fft2d(re, im, n, inverse = false) {
  const rr = new Float64Array(n)
  const ri = new Float64Array(n)
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) { rr[x] = re[y * n + x]; ri[x] = im[y * n + x] }
    fft1d(rr, ri, inverse)
    for (let x = 0; x < n; x++) { re[y * n + x] = rr[x]; im[y * n + x] = ri[x] }
  }
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) { rr[y] = re[y * n + x]; ri[y] = im[y * n + x] }
    fft1d(rr, ri, inverse)
    for (let y = 0; y < n; y++) { re[y * n + x] = rr[y]; im[y * n + x] = ri[y] }
  }
}

/**
 * A 1-D Cartesian undersampling mask: keep a fully-sampled centre, then random phase
 * encodes outside it. This is the `random1d` family the project's MRI grid uses.
 *
 * The centre is always kept because that is what real protocols do -- low frequencies carry
 * contrast and coil calibration -- and it is why even a heavily accelerated scan still looks
 * broadly right while losing fine structure.
 */
export function randomMask1d(n, acceleration, centerFraction = 0.08, seed = 0) {
  const mask = new Uint8Array(n)
  // **Natural FFT ordering: DC is index 0, not n/2.** The low frequencies -- the "centre"
  // of k-space that every protocol samples fully, because they carry contrast -- are the
  // indices nearest 0 *and* nearest n. Placing the centre block at n/2 instead samples the
  // highest frequencies fully and leaves DC out, which is both physically backwards and
  // not symmetric under i -> n - i. The verification residual sat at 2e-5 until this was
  // fixed, against 1e-15 after.
  const nCentre = Math.max(1, Math.round(n * centerFraction))
  const half = Math.floor(nCentre / 2)
  mask[0] = 1
  for (let i = 1; i <= half; i++) { mask[i] = 1; mask[n - i] = 1 }

  let kept = mask.reduce((s, v) => s + v, 0)
  const budget = Math.max(kept, Math.round(n / acceleration))
  let remaining = budget - kept
  // Deterministic PRNG, so the same controls give the same mask on every machine.
  let s = seed * 2654435761 + 1
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0
    return s / 4294967296
  }
  let guard = 0
  while (remaining > 0 && guard++ < n * 50) {
    const i = Math.floor(rand() * n)
    if (mask[i]) continue
    // **Conjugate-symmetric, and this is load-bearing.** The image is real, so its
    // spectrum satisfies F[-k] = conj(F[k]). Keeping line k without its partner -k
    // destroys that symmetry, and then P_R x and P_N x are complex even though x is real
    // -- so taking the real part for display silently leaves the null space, and the
    // invention stops being data-consistent. A verification run caught exactly that: the
    // k-space residual came back at 4.7e-2 instead of machine precision.
    //
    // Pairing the lines keeps both projectors real-to-real, which makes the demo's claim
    // exactly true rather than approximately. Real protocols exploit the same symmetry
    // (partial Fourier), so this is a simplification in the direction of the physics.
    const j = (n - i) % n
    if (j === i) { mask[i] = 1; remaining--; continue }  // self-paired (index 0, or n/2)
    mask[i] = 1
    mask[j] = 1
    remaining -= 2
  }
  return mask
}

/** Apply the mask along the phase-encode axis (rows of k-space). */
function applyMask(re, im, n, mask) {
  for (let y = 0; y < n; y++) {
    if (mask[y]) continue
    for (let x = 0; x < n; x++) { re[y * n + x] = 0; im[y * n + x] = 0 }
  }
}

/**
 * Split an image into what the operator measures and what it cannot see, and optionally
 * inject invented structure into the invisible half.
 *
 * Returns the two components, the recombined "reconstruction", and the k-space residual
 * between the original and the reconstruction **restricted to the measured lines** — which
 * is the number that matters: it stays at machine precision however much is invented.
 */
export function decompose(pixels, n, mask, hallucination = null) {
  const re = Float64Array.from(pixels)
  const im = new Float64Array(n * n)
  fft2d(re, im, n)

  const kRe = Float64Array.from(re)
  const kIm = Float64Array.from(im)

  // range: measured lines only
  const rRe = Float64Array.from(re)
  const rIm = Float64Array.from(im)
  applyMask(rRe, rIm, n, mask)
  const rangeK = { re: Float64Array.from(rRe), im: Float64Array.from(rIm) }
  fft2d(rRe, rIm, n, true)

  // null: everything else
  const nRe = new Float64Array(n * n)
  const nIm = new Float64Array(n * n)
  for (let i = 0; i < n * n; i++) {
    const y = Math.floor(i / n)
    if (!mask[y]) { nRe[i] = kRe[i]; nIm[i] = kIm[i] }
  }
  fft2d(nRe, nIm, n, true)

  // The invention: added in image space, then projected into the null space so that it is
  // data-consistent *by construction*. Projecting is the whole trick -- an arbitrary blob
  // would change the measured lines and be caught immediately.
  let invented = null
  let recon = new Float64Array(n * n)
  for (let i = 0; i < n * n; i++) recon[i] = rRe[i] + nRe[i]

  if (hallucination) {
    const hRe = Float64Array.from(hallucination)
    const hIm = new Float64Array(n * n)
    fft2d(hRe, hIm, n)
    for (let y = 0; y < n; y++) {
      if (!mask[y]) continue
      for (let x = 0; x < n; x++) { hRe[y * n + x] = 0; hIm[y * n + x] = 0 }
    }
    fft2d(hRe, hIm, n, true)
    invented = hRe
    for (let i = 0; i < n * n; i++) recon[i] += hRe[i]
  }

  return { range: rRe, nullSpace: nRe, recon, invented, rangeK, mask }
}

/**
 * Relative error between two images **on the measured k-space lines only**.
 *
 * This is the data-consistency check a reconstruction method would run. For anything added
 * inside the null space it returns ~1e-16: the invention is undetectable from the data.
 */
export function measurementError(a, b, n, mask) {
  const ar = Float64Array.from(a)
  const ai = new Float64Array(n * n)
  const br = Float64Array.from(b)
  const bi = new Float64Array(n * n)
  fft2d(ar, ai, n)
  fft2d(br, bi, n)
  let num = 0
  let den = 0
  for (let y = 0; y < n; y++) {
    if (!mask[y]) continue
    for (let x = 0; x < n; x++) {
      const i = y * n + x
      num += (ar[i] - br[i]) ** 2 + (ai[i] - bi[i]) ** 2
      den += ar[i] ** 2 + ai[i] ** 2
    }
  }
  return den > 0 ? Math.sqrt(num / den) : 0
}

/** A soft elliptical blob — the stand-in for invented anatomy. */
export function blob(n, cx, cy, rx, ry, amp) {
  const out = new Float64Array(n * n)
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const dx = (x - cx) / rx
      const dy = (y - cy) / ry
      const d = dx * dx + dy * dy
      if (d < 1) out[y * n + x] = amp * Math.exp(-3 * d) * (1 - d)
    }
  }
  return out
}
