/**
 * Split-conformal calibration, in the browser.
 *
 * This is a faithful port of `caliber.calibration.conformal.split_threshold` and the
 * coverage accounting around it — not an illustration of them. The site computes its
 * numbers from the exported score bank rather than displaying stored results, so moving
 * the alpha slider runs the actual method.
 *
 * Two details are carried over deliberately, because both change the answer:
 *
 *  - **Floor, not ceiling, on k.** `k = floor(alpha * (n + 1))`. Rounding up would return
 *    a threshold from as few as one positive: at n = 1, alpha = 0.1 it gives k = 1, so tau
 *    is that single score and the true miss rate is 1/2 by symmetry — five times the
 *    promise, reported as certified. Flooring sends thin groups to `certifiable: false`.
 *  - **Coverage is the pooled ratio** `caught / positives` over regions, matching the
 *    Python so the two agree on the marginal quantity.
 */

/** Threshold controlling the region-level miss rate at `alpha`, from calibration scores. */
export function splitThreshold(scores, labels, alpha) {
  const positives = []
  for (let i = 0; i < scores.length; i++) if (labels[i]) positives.push(scores[i])
  positives.sort((a, b) => a - b)
  const n = positives.length
  const k = Math.floor(alpha * (n + 1))
  if (k < 1) {
    // Cannot certify. -Infinity flags every region: trivially zero misses, so it is
    // valid, and useless in a way the caller can see and report.
    return { tau: -Infinity, nPositive: n, certifiable: false }
  }
  return { tau: positives[k - 1], nPositive: n, certifiable: true }
}

/** Coverage (caught / positives) and flagged area for a threshold on a set of rows. */
export function evaluate(scores, labels, tau) {
  let caught = 0
  let positives = 0
  let flagged = 0
  for (let i = 0; i < scores.length; i++) {
    const isFlag = scores[i] >= tau
    if (isFlag) flagged++
    if (labels[i]) {
      positives++
      if (isFlag) caught++
    }
  }
  return {
    coverage: positives ? caught / positives : NaN,
    flagArea: scores.length ? flagged / scores.length : NaN,
    nPositive: positives,
    nRegions: scores.length,
  }
}

/** Rows of the bank matching an operator index and split, as plain arrays. */
function slice(bank, opIndex, split) {
  const scores = []
  const labels = []
  for (let i = 0; i < bank.op.length; i++) {
    if (bank.op[i] === opIndex && bank.split[i] === split) {
      scores.push(bank.score[i])
      labels.push(bank.label[i])
    }
  }
  return { scores, labels }
}

/**
 * The central comparison, per deployment operator.
 *
 * `transferred` fits one threshold at the calibration operator and carries it everywhere.
 * `groupConditional` fits a separate threshold per operator. Both are fitted on the
 * **calib** split and scored on the **test** split, which are disjoint by sample — without
 * that the thresholds would be fitted on the rows they are scored on and every coverage
 * would land near nominal by construction, which is the artefact this project is about.
 */
export function compare(bank, alpha, calibrationIndex) {
  const away = slice(bank, calibrationIndex, 0)
  const transferredTau = splitThreshold(away.scores, away.labels, alpha)

  return bank.operators.map((name, i) => {
    const own = slice(bank, i, 0)
    const test = slice(bank, i, 1)
    const groupTau = splitThreshold(own.scores, own.labels, alpha)
    return {
      operator: name,
      distance: bank.distance[i],
      isCalibration: i === calibrationIndex,
      transferred: {
        ...evaluate(test.scores, test.labels, transferredTau.tau),
        tau: transferredTau.tau,
        certifiable: transferredTau.certifiable,
      },
      group: {
        ...evaluate(test.scores, test.labels, groupTau.tau),
        tau: groupTau.tau,
        certifiable: groupTau.certifiable,
      },
    }
  })
}

/** Spearman rank correlation, for the gap-against-distance trend. */
export function spearman(a, b) {
  const rank = (xs) => {
    const order = xs.map((v, i) => [v, i]).sort((p, q) => p[0] - q[0])
    const r = new Array(xs.length)
    let i = 0
    while (i < order.length) {
      let j = i
      while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j++
      const avg = (i + j) / 2 + 1
      for (let k = i; k <= j; k++) r[order[k][1]] = avg
      i = j + 1
    }
    return r
  }
  const ra = rank(a)
  const rb = rank(b)
  const n = a.length
  const mean = (xs) => xs.reduce((s, v) => s + v, 0) / xs.length
  const ma = mean(ra)
  const mb = mean(rb)
  let num = 0
  let da = 0
  let db = 0
  for (let i = 0; i < n; i++) {
    num += (ra[i] - ma) * (rb[i] - mb)
    da += (ra[i] - ma) ** 2
    db += (rb[i] - mb) ** 2
  }
  return da && db ? num / Math.sqrt(da * db) : NaN
}
