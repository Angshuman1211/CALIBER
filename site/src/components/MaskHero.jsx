import { useEffect, useRef, useState } from 'react'
import { prefersReducedMotion } from '../lib/motion'
import { randomMask1d } from '../lib/mri'

const N = 96

/**
 * The hero visual: k-space lines being discarded as acceleration rises.
 *
 * Decorative animation on a research page is noise, so this one carries the argument
 * instead. Each row is a phase-encode line; lit rows are measured, dark rows are not. As
 * the loop accelerates, most of the plane goes dark — and *that* dark region is the null
 * space, the place where a generative prior is free to invent structure that no
 * consistency check can contradict. The caption counts the lines so the ratio is legible
 * rather than impressionistic.
 *
 * It uses the same `randomMask1d` the live demo does, so the picture is the real sampling
 * pattern rather than an illustration of one.
 */
export default function MaskHero() {
  const canvasRef = useRef(null)
  const [accel, setAccel] = useState(2)
  const [kept, setKept] = useState(N)

  useEffect(() => {
    const reduced = prefersReducedMotion()
    const canvas = canvasRef.current
    if (!canvas) return undefined
    const ctx = canvas.getContext('2d')

    let raf = 0
    let t0 = performance.now()
    let current = reduced ? 6 : 2

    const draw = (R) => {
      const mask = randomMask1d(N, R, 0.08, 0)
      setKept(mask.reduce((s, v) => s + v, 0))
      setAccel(R)
      const css = getComputedStyle(document.documentElement)
      const lit = css.getPropertyValue('--accent').trim() || '#5ab0ff'
      const dark = css.getPropertyValue('--surface-2').trim() || '#181d28'
      ctx.clearRect(0, 0, N, N)
      for (let y = 0; y < N; y++) {
        ctx.fillStyle = mask[y] ? lit : dark
        ctx.globalAlpha = mask[y] ? 1 : 0.55
        ctx.fillRect(0, y, N, 1)
      }
      ctx.globalAlpha = 1
    }

    if (reduced) {
      draw(6)
      return undefined
    }

    const loop = (now) => {
      // 2 -> 12 and back, slowly, pausing at each end so the extremes register.
      const period = 7200
      const p = ((now - t0) % period) / period
      const tri = p < 0.5 ? p * 2 : (1 - p) * 2
      const eased = tri * tri * (3 - 2 * tri)
      const R = Math.max(2, Math.round(2 + eased * 10))
      if (R !== current) {
        current = R
        draw(R)
      }
      raf = requestAnimationFrame(loop)
    }
    draw(2)
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [])

  return (
    <div className="maskhero">
      <canvas
        ref={canvasRef}
        width={N}
        height={N}
        role="img"
        aria-label="k-space sampling lines, thinning as acceleration increases"
      />
      <div className="maskmeta">
        <div className="row">
          <span>acceleration</span>
          <span className="mono val">R={accel}</span>
        </div>
        <div className="row">
          <span>lines measured</span>
          <span className="mono val">
            {kept}/{N}
          </span>
        </div>
        <div className="row">
          <span>invisible to the scan</span>
          <span className="mono val bad">{(((N - kept) / N) * 100).toFixed(0)}%</span>
        </div>
      </div>
    </div>
  )
}
