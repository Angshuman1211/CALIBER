import { NOMINAL } from '../data'
import { useReveal } from '../lib/motion'

/**
 * Coverage against the nominal line, one row per deployment operator.
 *
 * Inline SVG rather than a charting library: one dependency fewer to install, to pin and
 * to break at deploy time, and the whole chart is twenty lines. Drawn as paired bars
 * against a dashed nominal rule, because the claim *is* the distance from that rule --
 * a y-axis starting anywhere but a fixed window would flatter or exaggerate it.
 */
export default function CoverageChart({ rows }) {
  const [wrapRef, shown] = useReveal()

  const W = 760
  const rowH = 30
  const padL = 96
  const padR = 56
  const top = 26
  const H = top + rows.length * rowH + 14
  const x0 = 0.6 // fixed window, so the fall is read at true scale
  const x1 = 1.0
  const sx = (v) => padL + ((v - x0) / (x1 - x0)) * (W - padL - padR)

  return (
    <div className="tablewrap chartwrap" ref={wrapRef}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        style={{ width: '100%', minWidth: 520, height: 'auto', display: 'block' }}
        role="img"
        aria-label="Empirical coverage by deployment operator, transferred versus group-conditional threshold"
      >
        {[0.6, 0.7, 0.8, 0.9, 1.0].map((t) => (
          <g key={t}>
            <line
              x1={sx(t)} x2={sx(t)} y1={top - 8} y2={H - 14}
              stroke="var(--border)"
              strokeDasharray={t === NOMINAL ? '4 3' : undefined}
            />
            <text
              x={sx(t)} y={top - 14} textAnchor="middle"
              fontSize="10" fill={t === NOMINAL ? 'var(--text)' : 'var(--faint)'}
              fontFamily="var(--mono)"
            >
              {t === NOMINAL ? '0.90 nominal' : t.toFixed(1)}
            </text>
          </g>
        ))}

        {rows.map((r, i) => {
          const y = top + i * rowH
          const below = r.naive < NOMINAL
          return (
            <g key={r.op}>
              <text x={padL - 10} y={y + 13} textAnchor="end" fontSize="12" fill="var(--muted)">
                {r.op}
              </text>
              {/* Bars grow from the axis rather than appearing, so the eye follows the
                  distance from the nominal rule — which is the quantity being claimed.
                  Transform-origin is the axis, not the bar, or they would grow from
                  their own left edge and land in the wrong place. */}
              <rect
                x={sx(x0)} y={y + 3} width={Math.max(sx(r.group) - sx(x0), 1)} height={7}
                fill="var(--accent)" opacity="0.85" rx="2"
                style={{
                  transform: shown ? 'scaleX(1)' : 'scaleX(0)',
                  transformOrigin: `${sx(x0)}px 0px`,
                  transition: `transform 0.75s var(--ease) ${0.05 + i * 0.05}s`,
                }}
              />
              <rect
                x={sx(x0)} y={y + 13} width={Math.max(sx(r.naive) - sx(x0), 1)} height={7}
                fill={below ? 'var(--bad)' : 'var(--warn)'} rx="2"
                style={{
                  transform: shown ? 'scaleX(1)' : 'scaleX(0)',
                  transformOrigin: `${sx(x0)}px 0px`,
                  transition: `transform 0.75s var(--ease) ${0.12 + i * 0.05}s`,
                }}
              />
              <text
                x={sx(Math.max(r.naive, r.group)) + 7} y={y + 15}
                fontSize="11" fill={below ? 'var(--bad)' : 'var(--muted)'} fontFamily="var(--mono)"
                style={{
                  opacity: shown ? 1 : 0,
                  transition: `opacity 0.4s ease ${0.5 + i * 0.05}s`,
                }}
              >
                {r.naive.toFixed(3)}
              </text>
            </g>
          )
        })}
      </svg>
      <p style={{ fontSize: 13, marginTop: 10 }}>
        <span style={{ color: 'var(--accent)' }}>■</span> group-conditional &nbsp;
        <span style={{ color: 'var(--bad)' }}>■</span> transferred, below nominal &nbsp;
        <span style={{ color: 'var(--warn)' }}>■</span> transferred, at nominal
      </p>
    </div>
  )
}
