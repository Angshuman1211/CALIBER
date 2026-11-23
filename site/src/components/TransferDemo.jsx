import { useMemo, useState } from 'react'
import bank from '../bank.json'
import { compare, spearman } from '../lib/conformal'

const pct = (v) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : '—')

export default function TransferDemo() {
  const [alpha, setAlpha] = useState(0.1)
  const [calIx, setCalIx] = useState(bank.operators.indexOf(bank.calibrationOperator))

  const rows = useMemo(() => compare(bank, alpha, calIx), [alpha, calIx])
  const nominal = 1 - alpha

  const trend = useMemo(() => {
    const d = rows.map((r) => r.distance)
    const gapT = rows.map((r) => nominal - r.transferred.coverage)
    const gapG = rows.map((r) => nominal - r.group.coverage)
    return { transferred: spearman(d, gapT), group: spearman(d, gapG) }
  }, [rows, nominal])

  const failing = rows.filter((r) => r.transferred.coverage < nominal - 1e-9).length
  const groupFailing = rows.filter((r) => r.group.coverage < nominal - 1e-9).length

  return (
    <div>
      <div className="card" style={{ marginBottom: 18 }}>
        <div className="controls">
          <label className="ctl">
            <span>
              Miss-rate budget <strong className="mono">α = {alpha.toFixed(2)}</strong> → promise{' '}
              <strong className="mono">{nominal.toFixed(2)}</strong>
            </span>
            <input
              type="range"
              min="0.02"
              max="0.30"
              step="0.01"
              value={alpha}
              onChange={(e) => setAlpha(Number(e.target.value))}
            />
          </label>
          <label className="ctl">
            <span>Calibrate at</span>
            <select value={calIx} onChange={(e) => setCalIx(Number(e.target.value))}>
              {bank.operators.map((o, i) => (
                <option key={o} value={i}>
                  {o}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div style={{ fontSize: 12, color: 'var(--faint)', marginTop: 10 }}>
          Thresholds are fitted on the <span className="mono">calib</span> split and scored on
          the <span className="mono">test</span> split, which share no sample. Computed in this
          tab from {bank.op.length.toLocaleString()} exported regions — moving the slider runs
          split conformal, it does not look up a stored answer.
        </div>
      </div>

      <div className="tablewrap">
        <table>
          <thead>
            <tr>
              <th>Deployed at</th>
              <th>Distance</th>
              <th>Transferred</th>
              <th>area</th>
              <th>Group-conditional</th>
              <th>area</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const bad = r.transferred.coverage < nominal - 1e-9
              const over = r.transferred.flagArea > 0.97
              return (
                <tr key={r.operator}>
                  <td>
                    {r.operator}
                    {r.isCalibration && (
                      <span className="pill" style={{ marginLeft: 8 }}>
                        calibrated here
                      </span>
                    )}
                  </td>
                  <td className="num" style={{ color: 'var(--faint)' }}>
                    {r.distance.toFixed(3)}
                  </td>
                  <td className={`num ${bad ? 'bad' : ''}`}>
                    {Number.isFinite(r.transferred.coverage)
                      ? r.transferred.coverage.toFixed(3)
                      : '—'}
                  </td>
                  <td className={`num ${over ? 'warn' : ''}`}>{pct(r.transferred.flagArea)}</td>
                  <td
                    className={`num ${
                      r.group.coverage < nominal - 1e-9 ? 'bad' : 'good'
                    }`}
                  >
                    {Number.isFinite(r.group.coverage) ? r.group.coverage.toFixed(3) : '—'}
                  </td>
                  <td className="num">{pct(r.group.flagArea)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="grid two" style={{ marginTop: 18 }}>
        <div className="card">
          <div style={{ fontSize: 13, color: 'var(--faint)' }}>
            Operators where the promise is broken
          </div>
          <div className="mono" style={{ fontSize: 24, marginTop: 4 }}>
            <span className={failing ? 'bad' : 'good'}>{failing}</span>
            <span style={{ color: 'var(--faint)' }}> transferred</span>
            <span style={{ color: 'var(--faint)', fontSize: 16 }}> · </span>
            <span className={groupFailing ? 'bad' : 'good'}>{groupFailing}</span>
            <span style={{ color: 'var(--faint)' }}> group-cond.</span>
          </div>
        </div>
        <div className="card">
          <div style={{ fontSize: 13, color: 'var(--faint)' }}>
            Gap vs operator distance (Spearman ρ)
          </div>
          <div className="mono" style={{ fontSize: 24, marginTop: 4 }}>
            {Number.isFinite(trend.transferred) ? trend.transferred.toFixed(2) : '—'}
            <span style={{ color: 'var(--faint)' }}> transferred</span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--faint)' }}>
            group-conditional{' '}
            {Number.isFinite(trend.group) ? trend.group.toFixed(2) : '—'} — a fix should show
            no trend
          </div>
        </div>
      </div>

      <div className="note" style={{ marginTop: 18 }}>
        <strong>Read the area column, not just coverage.</strong> On this synthetic bank the
        transferred threshold often keeps its promise by flagging nearly everything — coverage
        1.000 at 100% area tells a radiologist nothing. That is the failure in its benign
        direction. On real CT it fails the dangerous way instead: coverage falls to 0.693 while
        invented regions go unflagged. Both are the same defect — a threshold that no longer
        means what it was calibrated to mean.
      </div>

      <p style={{ fontSize: 12, color: 'var(--faint)' }}>
        Label: <span className="mono">{bank.labelText}</span>. Synthetic phantoms with a
        reference sampler, so the pipeline stamps this bank <span className="mono">NOT
        REPORTABLE</span> — it demonstrates the mechanism, it is not a result.
      </p>
    </div>
  )
}
