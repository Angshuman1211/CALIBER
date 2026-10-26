import { useEffect, useState } from 'react'

/**
 * Hash routing in about twenty lines.
 *
 * A router library would be the conventional choice, and for five static pages it would be
 * the larger decision: another dependency to pin, audit and keep current, for behaviour
 * that is `location.hash` plus a listener. Hash routes also deploy anywhere — GitHub Pages,
 * a university server directory, a file:// copy on a USB stick for a presentation — with no
 * server-side rewrite rules, which history routing needs and which is exactly the thing that
 * breaks ten minutes before a demo.
 */
export function useRoute(fallback) {
  const read = () => window.location.hash.replace(/^#\/?/, '') || fallback
  const [route, setRoute] = useState(read)

  useEffect(() => {
    const onChange = () => {
      setRoute(read())
      window.scrollTo({ top: 0 })
    }
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  })

  return route
}

export function navigate(to) {
  window.location.hash = `/${to}`
}
