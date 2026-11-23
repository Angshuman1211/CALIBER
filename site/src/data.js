// Every number the site shows, in one place, each with the document it came from.
//
// Nothing here is invented or rounded for effect. If a figure changes in the repository
// it changes here and nowhere else, which is the same discipline the analysis scripts use
// for label thresholds: one definition, so two copies cannot drift.
//
// IMPORTANT: every measured number below is **development output**. The banks were built
// with a natural-image prior applied to medical anatomy (D-016), so magnitudes will move
// when they are re-run under the medical priors. The *structure* -- monotone loss,
// restoration at every operator, the distance ordering -- is what is being shown.

export const NOMINAL = 0.9

/** CT, calibrated at 180 views, 13-split means. Source: T2.7, T2.8. */
export const CT_COVERAGE = [
  { op: '120 views', axis: 'views', naive: 0.890, group: 0.912, failsIn: null },
  { op: '90 views', axis: 'views', naive: 0.847, group: 0.915, failsIn: '13/13' },
  { op: '60 views', axis: 'views', naive: 0.770, group: 0.917, failsIn: '13/13' },
  { op: '45 views', axis: 'views', naive: 0.728, group: 0.884, failsIn: '13/13' },
  { op: '30 views', axis: 'views', naive: 0.693, group: 0.873, failsIn: '13/13' },
  { op: '150° arc', axis: 'arc', naive: 0.888, group: 0.920, failsIn: null },
  { op: '120° arc', axis: 'arc', naive: 0.837, group: 0.903, failsIn: null },
  { op: '90° arc', axis: 'arc', naive: 0.796, group: 0.855, failsIn: '13/13' },
]

/** What restoring coverage costs, single split. Source: T2.8 §2. */
export const RESTORATION_COST = [
  { op: '30 views', areaFrom: 0.220, areaTo: 0.313, fdrFrom: 0.029, fdrTo: 0.106 },
  { op: '90° arc', areaFrom: 0.308, areaTo: 0.423, fdrFrom: 0.084, fdrTo: 0.202 },
]

/**
 * The live demo, reproducible in seconds with no dataset and no checkpoint.
 * Source: `configs/experiments/demo_coverage_failure.yaml` + `analyse_bank.py`.
 * Synthetic phantoms, reference sampler — the harness stamps it NOT REPORTABLE.
 */
export const DEMO = [
  { op: 'R2 (calibrated here)', naiveCov: 0.9286, naiveArea: 0.3240, groupCov: 0.9286, groupArea: 0.3240 },
  { op: 'R3', naiveCov: 1.0, naiveArea: 0.9997, groupCov: 0.9061, groupArea: 0.5065 },
  { op: 'R4', naiveCov: 1.0, naiveArea: 1.0, groupCov: 0.9040, groupArea: 0.6792 },
  { op: 'R6', naiveCov: 1.0, naiveArea: 1.0, groupCov: 0.9020, groupArea: 0.6635 },
  { op: 'R8', naiveCov: 1.0, naiveArea: 1.0, groupCov: 0.9022, groupArea: 0.7794 },
]

export const DEMO_COMMANDS = [
  {
    label: 'Build a score bank, end to end',
    note: '3.5 s · 19,200 scored regions · no dataset, no checkpoint, no GPU',
    cmd: 'PYTHONPATH=src python3 -m caliber.runner \\\n    configs/experiments/demo_coverage_failure.yaml',
  },
  {
    label: 'Compare the three calibration methods',
    note: 'the table below is this command’s output',
    cmd: 'PYTHONPATH=src python3 scripts/analyse_bank.py runs/<run_id> \\\n    --calibration-operator mri.random1d.R2.c0.08.s0 --label ratio --tau 1.10',
  },
]

/** Rank correlation of the coverage gap with operator distance. Source: T2.7 §157, T2.8. */
export const DISTANCE_ORDERING = [
  { label: 'CT, signed null-space growth', rho: 0.976, good: true },
  { label: 'CT, symmetric measures', rho: 0.762, good: false },
  { label: 'MRI, from random1d R8', rho: 0.556, good: false },
  { label: 'MRI, from random1d R2', rho: 0.316, good: false },
]

export const SCALE = [
  { k: 'MRI bank', v: '1,791 pairs · 458,496 regions' },
  { k: 'CT bank', v: '900 pairs · 230,400 regions' },
  { k: 'Operator grid', v: '9 MRI · 9 CT' },
  { k: 'Patient splits', v: '13' },
  { k: 'Tests', v: '564 passing' },
]

/** Stated limits. Source: T1.7 §7, D67, D81. The honest half of the writeup. */
export const LIMITS = [
  {
    title: 'The detector does not beat a simpler baseline',
    body:
      'A posterior-spread baseline matches the operator-aware score to within 0.001–0.012 AUROC, ' +
      'because the solver already confines its spread to the null space. Measured, reported, not buried.',
  },
  {
    title: 'Detection numbers here measure an identity, not a detector',
    body:
      'From the decomposition, err² = s² + c² − 2·s·c·cos θ. Under a mismatched prior cos θ ≈ 0, so any ' +
      'criterion thresholding an error magnitude becomes a function of the score being graded — Spearman 0.99. ' +
      'Published AUROCs of 0.95–1.00 in this area, including ours, should be read with that in mind.',
  },
  {
    title: 'No exact guarantee under arbitrary operator change',
    body:
      'Exactness holds for a known, finite operator set. Continuous change targets a coverage-gap bound, not ' +
      'exact coverage, and operator shift is not covariate shift — so the textbook reweighting correction does not transfer.',
  },
  {
    title: 'Every number is development output',
    body:
      'The banks were built with a natural-image prior applied to medical anatomy. Medical priors are now wired ' +
      'and the banks are being re-run; the structure of the result is expected to hold, the magnitudes will move.',
  },
]

export const TERMS = [
  { term: 'Flagging', plain: 'Point at specific patches and say "don’t trust this bit."' },
  {
    term: 'Hallucination',
    plain:
      'The AI made that patch up. The scanner never measured it, so no consistency check against the scan data can catch it.',
  },
  {
    term: 'Conformal',
    plain:
      'A distribution-free guarantee on how often a flag is missed. Set 10%, catch at least 90% — whatever the model, whatever the patient.',
  },
  { term: 'Operator', plain: 'How the scan was taken: how many angles, how fast, which samples.' },
  { term: 'Shift', plain: 'The hospital changes that — a faster protocol, fewer angles.' },
  { term: 'Robust', plain: 'The guarantee still holds after they change it. By default, it does not.' },
]
