import { useEffect, useState } from 'react'
import CoverageChart from './components/CoverageChart'
import InvisibleDemo from './components/InvisibleDemo'
import TransferDemo from './components/TransferDemo'
import { useRoute } from './lib/router'
import {
  CT_COVERAGE,
  DEMO_COMMANDS,
  DISTANCE_ORDERING,
  LIMITS,
  NOMINAL,
  RESTORATION_COST,
  SCALE,
  TERMS,
} from './data'

const PAGES = [
  { id: 'overview', label: 'Overview' },
  { id: 'invisible', label: 'The invisible half' },
  { id: 'transfer', label: 'Watch it break' },
  { id: 'results', label: 'Results' },
  { id: 'limits', label: 'What we don’t claim' },
  { id: 'reproduce', label: 'Reproduce' },
]

const pct = (v) => `${(v * 100).toFixed(1)}%`

function Nav({ route }) {
  const [theme, setTheme] = useState(null)
  useEffect(() => {
    if (theme) document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  return (
    <header className="nav">
      <div className="wrap navinner">
        <a href="#/overview" className="brand">
          CALIBER
        </a>
        <nav>
          {PAGES.map((p) => (
            <a key={p.id} href={`#/${p.id}`} className={route === p.id ? 'on' : ''}>
              {p.label}
            </a>
          ))}
        </nav>
        <button
          className="themebtn"
          onClick={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
          aria-label="Toggle colour scheme"
        >
          {theme === 'light' ? '◐' : '◑'}
        </button>
      </div>
    </header>
  )
}

function Overview() {
  return (
    <>
      <section className="hero">
        <div className="wrap">
          <div className="eyebrow">Operator-shift-robust conformal hallucination flagging</div>
          <h1>
            Reliability guarantees for AI-reconstructed scans don’t survive a change of
            scanner settings.
          </h1>
          <p className="lead">
            Fast MRI and low-dose CT work by measuring less, and an AI prior fills in the
            rest. Sometimes it fills in anatomy that was never there — and because the
            invention sits exactly where the scanner never looked,{' '}
            <strong>no consistency check against the scan data can catch it</strong>.
          </p>
          <p className="lead">
            Conformal prediction can flag those regions with a guarantee: at most 10% missed.
            We show that guarantee silently breaks when the acquisition protocol changes,
            measure what it costs in two modalities, and restore it.
          </p>
          <div className="cta">
            <a className="btn primary" href="#/invisible">
              See it for yourself →
            </a>
            <a className="btn" href="#/results">
              The measured result
            </a>
          </div>
        </div>
      </section>

      <section>
        <div className="wrap">
          <h2>The title, word by word</h2>
          <ul className="clean">
            {TERMS.map((t) => (
              <li key={t.term}>
                <h3>{t.term}</h3>
                <p style={{ margin: 0 }}>{t.plain}</p>
              </li>
            ))}
          </ul>
          <div className="note">
            <strong>The load-bearing word is “shift”.</strong> Conformal prediction assumes
            calibration and deployment data are exchangeable. Changing the operator breaks
            that — and it is <em>not</em> covariate shift, because the measurement process
            itself changes rather than the inputs to a fixed one. The textbook reweighting
            correction does not apply, which is why the remedy here is conditioning.
          </div>
        </div>
      </section>

      <section>
        <div className="wrap">
          <h2>Scale</h2>
          <div className="grid two">
            {SCALE.map((s) => (
              <div className="card" key={s.k}>
                <div style={{ fontSize: 13, color: 'var(--faint)' }}>{s.k}</div>
                <div className="mono" style={{ fontSize: 16 }}>
                  {s.v}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </>
  )
}

function Invisible() {
  return (
    <section>
      <div className="wrap">
        <div className="eyebrow">Interactive · runs in your browser</div>
        <h2>The invisible half</h2>
        <p className="lead">
          Upload any image, or use the built-in phantom. We undersample it in k-space exactly
          as an accelerated MRI scan would, split it into what the scanner measures and what
          it cannot see, then invent a lesion inside the invisible half — and check whether
          the scan data can tell.
        </p>
        <InvisibleDemo />
      </div>
    </section>
  )
}

function Transfer() {
  return (
    <section>
      <div className="wrap">
        <div className="eyebrow">Interactive · real score bank</div>
        <h2>Watch the guarantee break</h2>
        <p className="lead">
          Calibrate a flag threshold at one acquisition setting, carry it to another, and see
          whether it keeps its promise. Split conformal runs live in this tab, on an exported
          bank of real scored regions.
        </p>
        <TransferDemo />
      </div>
    </section>
  )
}

function Results() {
  return (
    <>
      <section>
        <div className="wrap">
          <div className="eyebrow">The finding</div>
          <h2>A threshold calibrated at one protocol stops keeping its promise at another</h2>
          <p>
            CT, calibrated at 180 views, deployed as views thin and the arc narrows. Means over
            13 patient splits; nominal {NOMINAL.toFixed(2)}.
          </p>
          <CoverageChart rows={CT_COVERAGE} />
          <div className="tablewrap" style={{ marginTop: 20 }}>
            <table>
              <thead>
                <tr>
                  <th>Deployed at</th>
                  <th>Transferred</th>
                  <th>Group-conditional</th>
                  <th>Below nominal in</th>
                </tr>
              </thead>
              <tbody>
                {CT_COVERAGE.map((r) => (
                  <tr key={r.op}>
                    <td>{r.op}</td>
                    <td className={`num ${r.naive < NOMINAL ? 'bad' : ''}`}>
                      {r.naive.toFixed(3)}
                    </td>
                    <td className="num good">{r.group.toFixed(3)}</td>
                    <td className="num">{r.failsIn ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ marginTop: 16 }}>
            At the sparsest protocol the promise of 90% delivers <strong>69.3%</strong>, and it
            fails on <strong>13 of 13</strong> patient splits at the four most severe
            operators — so it is not sampling noise. Calibrating per operator restores
            0.873–0.920 everywhere.
          </p>
        </div>
      </section>

      <section>
        <div className="wrap">
          <h2>What the fix costs</h2>
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>Operator</th>
                  <th>Flagged area</th>
                  <th>Region FDR</th>
                </tr>
              </thead>
              <tbody>
                {RESTORATION_COST.map((r) => (
                  <tr key={r.op}>
                    <td>{r.op}</td>
                    <td className="num">
                      {pct(r.areaFrom)} → {pct(r.areaTo)}
                    </td>
                    <td className="num">
                      {pct(r.fdrFrom)} → {pct(r.fdrTo)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ marginTop: 14 }}>
            The transferred threshold’s near-zero false-discovery rate is not precision. It
            flags too little, and misses a quarter to a third of the invented regions.
          </p>
        </div>
      </section>

      <section>
        <div className="wrap">
          <h2>The failure is predictable before deployment</h2>
          <p>
            From the two forward operators alone — no ground truth, no reconstructions. Rank
            correlation with the measured coverage gap:
          </p>
          <div className="tablewrap">
            <table>
              <thead>
                <tr>
                  <th>Measure</th>
                  <th>ρ</th>
                </tr>
              </thead>
              <tbody>
                {DISTANCE_ORDERING.map((d) => (
                  <tr key={d.label}>
                    <td>{d.label}</td>
                    <td className={`num ${d.good ? 'good' : ''}`}>+{d.rho.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="note">
            <strong>On MRI it barely orders at all, and we say so.</strong> MRI’s worst
            failures sit at <em>matched</em> acceleration across different sampling patterns,
            where any scalar operator distance is near zero. So we do not claim the two
            modalities collapse onto one curve — only that the same failure and the same fix
            appear in both.
          </div>
        </div>
      </section>
    </>
  )
}

function Limits() {
  return (
    <section>
      <div className="wrap">
        <div className="eyebrow">Scope</div>
        <h2>What we don’t claim</h2>
        <p className="lead">
          Stating these precisely is part of the contribution, not a disclaimer attached to
          it.
        </p>
        <ul className="clean">
          {LIMITS.map((l) => (
            <li key={l.title}>
              <h3>{l.title}</h3>
              <p style={{ margin: 0 }}>{l.body}</p>
            </li>
          ))}
        </ul>
      </div>
    </section>
  )
}

function Reproduce() {
  return (
    <section>
      <div className="wrap">
        <div className="eyebrow">Reproduce</div>
        <h2>Five seconds, no dataset, no model weights</h2>
        <p>
          The phantom path needs no data access and no checkpoint, so the whole pipeline can
          be exercised on any machine.
        </p>
        {DEMO_COMMANDS.map((c, i) => (
          <div key={c.label} style={{ marginBottom: 18 }}>
            <h3>
              {i + 1}. {c.label}
            </h3>
            <p style={{ margin: '0 0 8px', fontSize: 13 }}>{c.note}</p>
            <pre>{c.cmd}</pre>
          </div>
        ))}
        <div className="note">
          <strong>The harness refuses to let these numbers be quoted.</strong> It stamps the
          run <span className="mono">NOT REPORTABLE</span> on its own, because the substrate
          is synthetic and the sampler is a reference rather than a generative prior. The same
          gate is what stops a mismatched prior’s output reaching the write-up.
        </div>
        <h3 style={{ marginTop: 28 }}>This site</h3>
        <pre>{'cd site\nnpm install\nnpm run dev     # http://localhost:5173\nnpm run build   # static dist/'}</pre>
      </div>
    </section>
  )
}

export default function App() {
  const route = useRoute('overview')
  const page =
    {
      overview: <Overview />,
      invisible: <Invisible />,
      transfer: <Transfer />,
      results: <Results />,
      limits: <Limits />,
      reproduce: <Reproduce />,
    }[route] ?? <Overview />

  return (
    <>
      <Nav route={route} />
      <main>{page}</main>
      <footer>
        <div className="wrap">
          CALIBER — conformal hallucination flagging under acquisition-operator shift. Every
          measured figure shown is development output: the banks were built with a
          natural-image prior applied to medical anatomy, and are being re-run under medical
          priors. The structure of the result is what is being shown; the magnitudes will
          move.
        </div>
      </footer>
    </>
  )
}
