/**
 * Gates for the site's two computed demos. `npm test`.
 *
 * The site does not display stored results — it runs split conformal and a 2-D FFT in the
 * browser — so its arithmetic needs the same treatment as the Python: checked against
 * properties that would break if it were wrong, with a control that must fail.
 *
 * Writing these caught two real bugs in the image demo, both of which produced a plausible
 * looking picture and a wrong number:
 *
 *   1. A non-conjugate-symmetric mask makes P_N x complex for real x, so taking the real
 *      part for display silently leaves the null space. Residual 4.7e-2 instead of 1e-15.
 *   2. The fully-sampled "centre" was placed at index n/2. In natural FFT ordering DC is
 *      index 0, so that sampled the *highest* frequencies fully and omitted DC — backwards
 *      physically, and not symmetric under i -> n - i. Residual 2.0e-5.
 *
 * Neither was visible by looking at the images.
 */

import assert from 'node:assert/strict'
import test from 'node:test'
import { blob, decompose, fft2d, measurementError, randomMask1d } from '../src/lib/mri.js'
import { compare, evaluate, splitThreshold } from '../src/lib/conformal.js'
import bank from '../src/bank.json' with { type: 'json' }

const N = 64
const EXACT = 1e-12

const ramp = () => Float64Array.from({ length: N * N }, (_, i) => ((i * 2654435761) % 1000) / 1000)

test('the FFT round-trips to machine precision', () => {
  const re = Float64Array.from({ length: N * N }, (_, i) => Math.sin(i * 0.37) + 0.3)
  const im = new Float64Array(N * N)
  const orig = Float64Array.from(re)
  fft2d(re, im, N)
  fft2d(re, im, N, true)
  let err = 0
  for (let i = 0; i < N * N; i++) err = Math.max(err, Math.abs(re[i] - orig[i]))
  assert.ok(err < EXACT, `round-trip error ${err}`)
})

test('the mask is conjugate-symmetric, so both projectors are real-to-real', () => {
  for (const R of [2, 3, 4, 6, 8, 12]) {
    const mask = randomMask1d(N, R, 0.08, 0)
    for (let i = 0; i < N; i++) {
      assert.equal(mask[i], mask[(N - i) % N], `R=${R}: line ${i} unpaired`)
    }
    assert.equal(mask[0], 1, 'DC must always be measured')
  }
})

test('range and null partition the image exactly', () => {
  const px = ramp()
  const mask = randomMask1d(N, 4, 0.08, 0)
  const out = decompose(px, N, mask, null)
  let err = 0
  for (let i = 0; i < N * N; i++) err = Math.max(err, Math.abs(out.recon[i] - px[i]))
  assert.ok(err < EXACT, `P_R x + P_N x differs from x by ${err}`)
})

test('THE CLAIM: invention in the null space leaves measured k-space untouched', () => {
  const px = ramp()
  const mask = randomMask1d(N, 4, 0.08, 0)
  const out = decompose(px, N, mask, blob(N, 40, 30, 7, 6, 0.6))

  const residual = measurementError(px, out.recon, N, mask)
  assert.ok(residual < 1e-10, `data-consistency residual ${residual} is not machine precision`)

  let visible = 0
  for (let i = 0; i < N * N; i++) visible = Math.max(visible, Math.abs(out.recon[i] - px[i]))
  assert.ok(visible > 0.05, `the invention must be visible in the image, got ${visible}`)
})

test('CONTROL: an unprojected blob is detectable, so the test above can fail', () => {
  const px = ramp()
  const mask = randomMask1d(N, 4, 0.08, 0)
  const h = blob(N, 40, 30, 7, 6, 0.6)
  const naive = Float64Array.from(px)
  for (let i = 0; i < N * N; i++) naive[i] += h[i]
  const residual = measurementError(px, naive, N, mask)
  assert.ok(residual > 1e-3, `an arbitrary blob must change the measurements, got ${residual}`)
})

test('split conformal controls the miss rate at alpha', () => {
  let s = 12345
  const rand = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296)
  const scores = []
  const labels = []
  for (let i = 0; i < 4000; i++) {
    const pos = rand() < 0.3
    labels.push(pos ? 1 : 0)
    scores.push((pos ? 2 : 0) + (rand() * 2 - 1) * 1.5)
  }
  for (const alpha of [0.05, 0.1, 0.2]) {
    const { tau } = splitThreshold(scores, labels, alpha)
    const { coverage } = evaluate(scores, labels, tau)
    assert.ok(coverage >= 1 - alpha - 0.01, `alpha=${alpha}: coverage ${coverage}`)
  }
})

test('a group too thin to certify returns -inf rather than a threshold from one point', () => {
  const r = splitThreshold([1, 2, 3], [1, 0, 0], 0.1)
  assert.equal(r.certifiable, false)
  assert.equal(r.tau, -Infinity)
  assert.equal(evaluate([1, 2, 3], [1, 0, 0], r.tau).coverage, 1)
})

test('the bundled bank keeps calibration and test samples disjoint', () => {
  const calib = new Set()
  const test_ = new Set()
  for (let i = 0; i < bank.op.length; i++) {
    ;(bank.split[i] === 0 ? calib : test_).add(bank.sample[i])
  }
  const shared = [...calib].filter((s) => test_.has(s))
  assert.equal(shared.length, 0, `${shared.length} samples appear on both sides`)
})

test('the transferred threshold loses what the group-conditional one keeps', () => {
  const rows = compare(bank, 0.1, bank.operators.indexOf(bank.calibrationOperator))
  assert.equal(rows.length, bank.operators.length)
  const away = rows.filter((r) => !r.isCalibration)
  // The demo's whole point: at least one deployment operator must be worse off under the
  // transferred threshold than under its own. If this ever passes trivially, the exported
  // bank has stopped showing the phenomenon and the page is misleading.
  const worse = away.filter(
    (r) => r.transferred.coverage < r.group.coverage - 1e-9 || r.transferred.flagArea > r.group.flagArea + 1e-9
  )
  assert.ok(worse.length > 0, 'no operator shows a transfer penalty in coverage or area')
})
