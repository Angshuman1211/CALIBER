import { useCallback, useEffect, useRef, useState } from 'react'
import { blob, decompose, measurementError, randomMask1d } from '../lib/mri'

const N = 128 // power of two, for the radix-2 FFT

/** Draw a Float64Array into a canvas, auto-scaled, optionally signed (blue/red). */
function paint(canvas, data, { signed = false } = {}) {
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  const img = ctx.createImageData(N, N)
  let lo = Infinity
  let hi = -Infinity
  for (const v of data) {
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  const mag = Math.max(Math.abs(lo), Math.abs(hi)) || 1
  for (let i = 0; i < N * N; i++) {
    const v = data[i]
    let r
    let g
    let b
    if (signed) {
      const t = Math.max(-1, Math.min(1, v / mag))
      // diverging: cool for negative, warm for positive, mid-grey at zero
      r = Math.round(128 + 127 * Math.max(0, t))
      g = Math.round(128 - 90 * Math.abs(t))
      b = Math.round(128 + 127 * Math.max(0, -t))
    } else {
      const t = hi > lo ? (v - lo) / (hi - lo) : 0
      r = g = b = Math.round(255 * Math.max(0, Math.min(1, t)))
    }
    img.data[i * 4] = r
    img.data[i * 4 + 1] = g
    img.data[i * 4 + 2] = b
    img.data[i * 4 + 3] = 255
  }
  ctx.putImageData(img, 0, 0)
}

/** The built-in phantom, so the demo works before anything is uploaded. */
function phantomPixels() {
  const out = new Float64Array(N * N)
  const ell = (cx, cy, rx, ry, val) => {
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const dx = (x - cx) / rx
        const dy = (y - cy) / ry
        if (dx * dx + dy * dy <= 1) out[y * N + x] = val
      }
    }
  }
  ell(64, 64, 54, 44, 0.3)
  ell(64, 64, 49, 39, 0.55)
  ell(46, 52, 14, 16, 0.8)
  ell(84, 74, 12, 11, 0.35)
  ell(70, 44, 9, 8, 0.95)
  return out
}

function Panel({ title, sub, canvasRef, signed, highlight }) {
  return (
    <figure style={{ margin: 0 }}>
      <canvas
        ref={canvasRef}
        width={N}
        height={N}
        style={{
          width: '100%',
          imageRendering: 'pixelated',
          borderRadius: 8,
          border: `1px solid ${highlight ? 'var(--warn)' : 'var(--border)'}`,
          display: 'block',
          background: 'var(--surface-2)',
          aspectRatio: '1 / 1',
        }}
      />
      <figcaption style={{ marginTop: 8 }}>
        <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>{title}</div>
        <div style={{ fontSize: 12, color: 'var(--faint)' }}>{sub}</div>
      </figcaption>
    </figure>
  )
}

/**
 * Shipped samples, so the demo is usable without finding an image first.
 *
 * The brain slices are real: BraTS 2021 test images, the same data Song et al.'s medical
 * prior was trained on, bundled from their released test set. The phantoms come from the
 * project's own `PhantomDataset`, so the synthetic path shown here is the one the pipeline
 * actually runs on.
 */
const SAMPLES = [
  { file: 'brain-1.png', label: 'Brain 1', kind: 'BraTS' },
  { file: 'brain-2.png', label: 'Brain 2', kind: 'BraTS' },
  { file: 'brain-3.png', label: 'Brain 3', kind: 'BraTS' },
  { file: 'brain-4.png', label: 'Brain 4', kind: 'BraTS' },
  { file: 'phantom-1.png', label: 'Phantom 1', kind: 'synthetic' },
  { file: 'phantom-2.png', label: 'Phantom 2', kind: 'synthetic' },
]

/** Draw any image source into an N x N luminance buffer. */
function toPixels(img) {
  const c = document.createElement('canvas')
  c.width = N
  c.height = N
  const ctx = c.getContext('2d')
  ctx.drawImage(img, 0, 0, N, N)
  const d = ctx.getImageData(0, 0, N, N).data
  const out = new Float64Array(N * N)
  for (let i = 0; i < N * N; i++) {
    // luminance; the project reconstructs magnitude images, which are single-channel
    out[i] = (0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2]) / 255
  }
  return out
}

export default function InvisibleDemo() {
  const [accel, setAccel] = useState(4)
  const [inject, setInject] = useState(true)
  const [strength, setStrength] = useState(0.55)
  const [pixels, setPixels] = useState(() => phantomPixels())
  const [source, setSource] = useState('built-in phantom')
  const [active, setActive] = useState('built-in')
  const [stats, setStats] = useState(null)
  const [busy, setBusy] = useState(false)

  const loadSample = (s) => {
    setBusy(true)
    const img = new Image()
    img.onload = () => {
      setPixels(toPixels(img))
      setSource(`${s.label} · ${s.kind}`)
      setActive(s.file)
      setBusy(false)
    }
    img.onerror = () => setBusy(false)
    // `base: './'` in vite.config.js, so a relative URL survives a subdirectory deploy.
    img.src = `${import.meta.env.BASE_URL}samples/${s.file}`
  }

  const origRef = useRef(null)
  const rangeRef = useRef(null)
  const nullRef = useRef(null)
  const reconRef = useRef(null)

  const run = useCallback(() => {
    const mask = randomMask1d(N, accel, 0.08, 0)
    const hall = inject ? blob(N, 78, 56, 13, 11, strength) : null
    const out = decompose(pixels, N, mask, hall)
    paint(origRef.current, pixels)
    paint(rangeRef.current, out.range)
    paint(nullRef.current, out.nullSpace, { signed: true })
    paint(reconRef.current, out.recon)
    const err = measurementError(pixels, out.recon, N, mask)
    const kept = mask.reduce((s, v) => s + v, 0)
    setStats({ err, kept, total: N, frac: kept / N })
  }, [pixels, accel, inject, strength])

  useEffect(() => {
    run()
  }, [run])

  const onFile = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    setBusy(true)
    const url = URL.createObjectURL(file)
    const img = new Image()
    img.onload = () => {
      setPixels(toPixels(img))
      setSource(file.name)
      setActive('upload')
      setBusy(false)
      URL.revokeObjectURL(url)
    }
    img.onerror = () => {
      setBusy(false)
      URL.revokeObjectURL(url)
    }
    img.src = url
  }

  return (
    <div>
      <div className="card" style={{ marginBottom: 18 }}>
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 13, color: 'var(--muted)', marginBottom: 8 }}>
            Try a sample, or upload your own below
          </div>
          <div className="samples">
            {SAMPLES.map((s) => (
              <button
                key={s.file}
                className={`sample${active === s.file ? ' on' : ''}`}
                onClick={() => loadSample(s)}
              >
                <img
                  src={`${import.meta.env.BASE_URL}samples/${s.file}`}
                  alt={s.label}
                  loading="lazy"
                />
                <span>{s.label}</span>
              </button>
            ))}
            <button
              className={`sample${active === 'built-in' ? ' on' : ''}`}
              onClick={() => {
                setPixels(phantomPixels())
                setSource('built-in phantom')
                setActive('built-in')
              }}
            >
              <span className="generated">generated</span>
              <span>Built-in</span>
            </button>
          </div>
        </div>
        <div className="controls">
          <label className="ctl">
            <span>Your own image</span>
            <input type="file" accept="image/*" onChange={onFile} />
          </label>
          <label className="ctl">
            <span>
              Acceleration <strong className="mono">R={accel}</strong>
            </span>
            <input
              type="range"
              min="2"
              max="12"
              step="1"
              value={accel}
              onChange={(e) => setAccel(Number(e.target.value))}
            />
          </label>
          <label className="ctl">
            <span>
              Invented structure{' '}
              <strong className="mono">{inject ? strength.toFixed(2) : 'off'}</strong>
            </span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={inject ? strength : 0}
              onChange={(e) => {
                const v = Number(e.target.value)
                setInject(v > 0)
                setStrength(v || 0.05)
              }}
            />
          </label>
        </div>
        <div style={{ fontSize: 12, color: 'var(--faint)', marginTop: 10 }}>
          Source: <span className="mono">{busy ? 'loading…' : source}</span> · resampled to{' '}
          {N}×{N}, luminance only. Nothing leaves your browser — the FFT runs in this tab.
          The brain samples are real BraTS 2021 slices, from the test set released with the
          medical prior this project uses.
        </div>
      </div>

      <div className="panels">
        <Panel canvasRef={origRef} title="Ground truth" sub="the full image" />
        <Panel
          canvasRef={rangeRef}
          title="What the scanner measures"
          sub={stats ? `${stats.kept}/${stats.total} k-space lines kept` : ''}
        />
        <Panel
          canvasRef={nullRef}
          title="What it cannot see"
          sub="the null space — invisible to the data"
          signed
        />
        <Panel
          canvasRef={reconRef}
          title={inject ? 'Reconstruction, with invention' : 'Reconstruction'}
          sub={inject ? 'a lesion that was never there' : 'no invention added'}
          highlight={inject}
        />
      </div>

      {stats && (
        <div className="card" style={{ marginTop: 18 }}>
          <div style={{ fontSize: 13, color: 'var(--faint)' }}>
            Data-consistency check — relative error on the <em>measured</em> k-space lines,
            between the ground truth and the reconstruction
          </div>
          <div
            className="bignum"
            style={{
              color: stats.err < 1e-10 ? 'var(--bad)' : 'var(--text)',
              marginTop: 4,
              transition: 'color 0.4s var(--ease)',
            }}
          >
            {stats.err < 1e-10 ? stats.err.toExponential(1) : stats.err.toFixed(6)}
          </div>
          <p style={{ margin: '8px 0 0' }}>
            {inject ? (
              <>
                <strong>That is machine precision, with a lesion added.</strong> The invention
                lives entirely in the null space, so every measured line is bit-identical.
                No consistency check against the scan data can detect it — which is why
                hallucination in a generative reconstruction needs a different kind of
                guarantee, and why this project exists.
              </>
            ) : (
              <>
                Zero, as expected with nothing invented. Now drag the{' '}
                <strong>invented structure</strong> slider up and watch this number stay at
                machine precision while a lesion appears in the reconstruction.
              </>
            )}
          </p>
        </div>
      )}
    </div>
  )
}
