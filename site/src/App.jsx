import { useEffect, useState } from 'react'
import CoverageChart from './components/CoverageChart'
import InvisibleDemo from './components/InvisibleDemo'
import MaskHero from './components/MaskHero'
import TransferDemo from './components/TransferDemo'
import { useCountUp, useReveal } from './lib/motion'
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

/** A block that fades up the first time it scrolls into view. */
function Reveal({ children, delay = 0, as: Tag = 'div', className = '', ...rest }) {
  const [ref, shown] = useReveal()
  const d = delay ? ` d${delay}` : ''
  return (
    <Tag ref={ref} className={`reveal${shown ? ` in${d}` : ''} ${className}`.trim()} {...rest}>
      {children}
    </Tag>
  )
}

function Nav({ route }) {
  const [theme, setTheme] = useState(null)
  useEffect(() => {
    if (theme) document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  return (
    <header className="nav">
      <div className="wrap navinner">
        <a href="#/overview" className="brand">
          <span className="dot" />
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

/** The headline number, counted up when it first appears. */
function Headline() {
  const [ref, v] = useCountUp(0.693, { start: 0.9, duration: 1400 })
  return (
    <div className="card headline" ref={ref}>
      <div className="k">Promised 90% coverage. At the sparsest CT protocol, delivered</div>
      <div className="bignum bad">{(v * 100).toFixed(1)}%</div>
      <div className="k">
        and it failed on <strong>13 of 13</strong> patient splits — so it is not noise
      </div>
    </div>
  )
}

function Overview() {
  return (
    <>
      <section className="hero">
        <div className="wrap herogrid">
          <div>
            <Reveal className="eyebrow">
              Operator-shift-robust conformal hallucination flagging
            </Reveal>
            <Reveal as="h1" delay={1}>
              Reliability guarantees for AI-reconstructed scans don’t survive a change of
              scanner settings.
            </Reveal>
            <Reveal delay={2}>
              <p className="lead">
                Fast MRI and low-dose CT work by measuring less, and an AI prior fills in the
                rest. Sometimes it fills in anatomy that was never there — and because the
                invention sits exactly where the scanner never looked,{' '}
                <strong>no consistency check against the scan data can catch it</strong>.
              </p>
              <p className="lead">
                Conformal prediction flags those regions with a guarantee: at most 10% missed.
                We show that guarantee silently breaks when the protocol changes, measure what
                it costs in two modalities, and restore it.
              </p>
              <div className="cta">
                <a className="btn primary" href="#/invisible">
                  See it for yourself <span className="arr">→</span>
                </a>
                <a className="btn" href="#/results">
                  The measured result
                </a>
              </div>
            </Reveal>
          </div>
          <Reveal delay={2}>
            <MaskHero />
          </Reveal>
        </div>
      </section>

      <section>
        <div className="wrap">
          <Reveal>
            <Headline />
          </Reveal>
        </div>
      </section>

      <section>
        <div className="wrap">
          <Reveal className="sec-head">
            <div className="eyebrow">Plain language</div>
            <h2>The title, word by word</h2>
          </Reveal>
          <ul className="clean">
            {TERMS.map((t, i) => (
              <Reveal as="li" key={t.term} delay={Math.min(3, i)}>
                <h3>{t.term}</h3>
                <p style={{ margin: 0 }}>{t.plain}</p>
              </Reveal>
            ))}
          </ul>
          <Reveal className="note">
            <strong>The load-bearing word is “shift”.</strong> Conformal prediction assumes
            calibration and deployment data are exchangeable. Changing the operator breaks
            that — and it is <em>not</em> covariate shift, because the measurement process
            itself changes rather than the inputs to a fixed one. The textbook reweighting
            correction does not apply, which is why the remedy here is conditioning.
          </Reveal>
        </div>
      </section>

      <section>
        <div className="wrap">
          <Reveal className="sec-head">
            <div className="eyebrow">Scale</div>
            <h2>What it was measured on</h2>
          </Reveal>
          <div className="grid three">
            {SCALE.map((s, i) => (
              <Reveal key={s.k} delay={Math.min(3, i)}>
                <div className="card hover stat">
                  <div className="k">{s.k}</div>
                  <div className="v">{s.v}</div>
                </div>
              </Reveal>
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
        <Reveal className="sec-head">
          <div className="eyebrow">Interactive · computed in your browser</div>
          <h2>The invisible half</h2>
          <p className="lead">
            Pick a sample or upload any image. It is undersampled in k-space exactly as an
            accelerated MRI scan would be, split into what the scanner measures and what it
            cannot see, then a lesion is invented inside the invisible half — and we run the
            data-consistency check a reconstruction method would run.
          </p>
        </Reveal>
        <Reveal delay={1}>
          <InvisibleDemo />
        </Reveal>
      </div>
    </section>
  )
}

function Transfer() {
  return (
    <section>
      <div className="wrap">
        <Reveal className="sec-head">
          <div className="eyebrow">Interactive · real score bank</div>
          <h2>Watch the guarantee break</h2>
          <p className="lead">
            Calibrate a flag threshold at one acquisition setting, carry it to another, and see
            whether it keeps its promise. Split conformal runs live in this tab, on an exported
            bank of real scored regions.
          </p>
        </Reveal>
        <Reveal delay={1}>
          <TransferDemo />
        </Reveal>
      </div>
    </section>
  )
}

function Results() {
  return (
    <>
      <section>
        <div className="wrap">
          <Reveal className="sec-head">
            <div className="eyebrow">The finding</div>
            <h2>A threshold calibrated at one protocol stops keeping its promise at another</h2>
            <p>
              CT, calibrated at 180 views, deployed as views thin and the arc narrows. Means
              over 13 patient splits; nominal {NOMINAL.toFixed(2)}.
            </p>
          </Reveal>
          <Reveal delay={1}>
            <CoverageChart rows={CT_COVERAGE} />
          </Reveal>
          <Reveal delay={2}>
            <div className="tablewrap" style={{ marginTop: 24 }}>
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
            <p style={{ marginTop: 18 }}>
              At the sparsest protocol the promise of 90% delivers <strong>69.3%</strong>, and
              it fails on <strong>13 of 13</strong> patient splits at the four most severe
              operators. Calibrating per operator restores 0.873–0.920 everywhere.
            </p>
          </Reveal>
        </div>
      </section>

      <section>
        <div className="wrap">
          <Reveal className="sec-head">
            <h2>What the fix costs</h2>
          </Reveal>
          <Reveal delay={1}>
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
            <p style={{ marginTop: 16 }}>
              The transferred threshold’s near-zero false-discovery rate is not precision. It
              flags too little, and misses a quarter to a third of the invented regions.
            </p>
          </Reveal>
        </div>
      </section>

      <section>
        <div className="wrap">
          <Reveal className="sec-head">
            <h2>The failure is predictable before deployment</h2>
            <p>
              From the two forward operators alone — no ground truth, no reconstructions. Rank
              correlation with the measured coverage gap:
            </p>
          </Reveal>
          <Reveal delay={1}>
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
          </Reveal>
          <Reveal className="note" delay={2}>
            <strong>On MRI it barely orders at all, and we say so.</strong> MRI’s worst
            failures sit at <em>matched</em> acceleration across different sampling patterns,
            where any scalar operator distance is near zero. So we do not claim the two
            modalities collapse onto one curve — only that the same failure and the same fix
            appear in both.
          </Reveal>
        </div>
      </section>
    </>
  )
}

function Limits() {
  return (
    <section>
      <div className="wrap">
        <Reveal className="sec-head">
          <div className="eyebrow">Scope</div>
          <h2>What we don’t claim</h2>
          <p className="lead">
            Stating these precisely is part of the contribution, not a disclaimer attached to
            it.
          </p>
        </Reveal>
        <ul className="clean">
          {LIMITS.map((l, i) => (
            <Reveal as="li" key={l.title} delay={Math.min(3, i)}>
              <h3>{l.title}</h3>
              <p style={{ margin: 0 }}>{l.body}</p>
            </Reveal>
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
        <Reveal className="sec-head">
          <div className="eyebrow">Reproduce</div>
          <h2>Five seconds, no dataset, no model weights</h2>
          <p>
            The phantom path needs no data access and no checkpoint, so the whole pipeline can
            be exercised on any machine.
          </p>
        </Reveal>
        {DEMO_COMMANDS.map((c, i) => (
          <Reveal key={c.label} delay={Math.min(3, i + 1)} style={{ marginBottom: 20 }}>
            <h3>
              {i + 1}. {c.label}
            </h3>
            <p style={{ margin: '0 0 9px', fontSize: 13 }}>{c.note}</p>
            <pre>{c.cmd}</pre>
          </Reveal>
        ))}
        <Reveal className="note">
          <strong>The harness refuses to let these numbers be quoted.</strong> It stamps the
          run <span className="mono">NOT REPORTABLE</span> on its own, because the substrate is
          synthetic and the sampler is a reference rather than a generative prior. The same
          gate is what stops a mismatched prior’s output reaching the write-up.
        </Reveal>
        <Reveal>
          <h3 style={{ marginTop: 30 }}>This site</h3>
          <pre>
            {'cd site\nnpm install\nnpm run dev     # http://localhost:5173\nnpm run build   # static dist/\nnpm test        # 9 gates on the maths the demos depend on'}
          </pre>
        </Reveal>
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
      {/* `key` remounts on navigation, which replays the entry animation and resets any
          demo state — a stale canvas from the previous page would be worse than a reset. */}
      <main key={route} className="pageturn">
        {page}
      </main>
      <footer>
        <div className="wrap">
          CALIBER — conformal hallucination flagging under acquisition-operator shift. Every
          measured figure shown is development output: the banks were built with a
          natural-image prior applied to medical anatomy, and are being re-run under medical
          priors. The structure of the result is what is shown; the magnitudes will move.
        </div>
      </footer>
    </>
  )
}
