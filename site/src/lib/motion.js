import { useEffect, useRef, useState } from 'react'

/** Honour the OS setting. Checked at call time, not cached, so it tracks live changes. */
export function prefersReducedMotion() {
  return (
    typeof window !== 'undefined' &&
    window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
  )
}

/**
 * Reveal-on-scroll, via IntersectionObserver.
 *
 * Fires once and then disconnects: content that re-animates every time it scrolls past is
 * distracting rather than polished. Under reduced motion it marks visible immediately, so
 * nothing is ever hidden behind an animation that will not play.
 */
export function useReveal(options = {}) {
  const ref = useRef(null)
  const [shown, setShown] = useState(() => prefersReducedMotion())

  useEffect(() => {
    if (shown || !ref.current) return undefined
    if (typeof IntersectionObserver === 'undefined') {
      setShown(true)
      return undefined
    }
    const el = ref.current
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setShown(true)
          io.disconnect()
        }
      },
      { rootMargin: options.rootMargin ?? '0px 0px -12% 0px', threshold: options.threshold ?? 0.08 },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [shown, options.rootMargin, options.threshold])

  return [ref, shown]
}

/**
 * Count a number up when it first appears.
 *
 * Eased rather than linear, so it decelerates into the final value instead of stopping
 * dead — and it always lands on exactly `to`, because a counter that settles on 0.6929
 * when the figure is 0.693 undermines the thing it is meant to emphasise.
 */
export function useCountUp(to, { duration = 1100, start = 0 } = {}) {
  const [ref, shown] = useReveal()
  const [value, setValue] = useState(prefersReducedMotion() ? to : start)

  useEffect(() => {
    if (!shown || prefersReducedMotion()) {
      setValue(to)
      return undefined
    }
    let raf = 0
    const t0 = performance.now()
    const tick = (now) => {
      const p = Math.min(1, (now - t0) / duration)
      const eased = 1 - (1 - p) ** 3
      setValue(start + (to - start) * eased)
      if (p < 1) raf = requestAnimationFrame(tick)
      else setValue(to)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [shown, to, duration, start])

  return [ref, value]
}
