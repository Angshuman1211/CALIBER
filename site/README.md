# CALIBER — project site

A six-page React site for showing the project, built around **two demos that compute rather
than recite**:

- **The invisible half** (`#/invisible`) — upload any image. It is undersampled in k-space
  exactly as an accelerated MRI scan would be, split into what the scanner measures and what
  it cannot see, and a lesion is injected into the invisible half. The page then runs the
  data-consistency check a reconstruction method would run, and reports **~1e-16**: the
  invention is undetectable from the measurements. Real radix-2 FFT, in the browser, nothing
  uploaded anywhere.
- **Watch it break** (`#/transfer`) — split conformal on an exported bank of 10,240 scored
  regions. Move the alpha slider or change the calibration operator and the thresholds are
  refitted live. It is a port of `caliber.calibration.conformal.split_threshold`, floor rule
  and all, not a lookup of stored answers.

```bash
cd site
npm install
npm run dev        # http://localhost:5173
npm run build      # -> dist/, static, no server needed
npm run preview    # serve dist/ locally to check the build
npm test           # 9 gates on the maths both demos depend on
```

Node 22, React 18, Vite 8. Two runtime dependencies (`react`, `react-dom`) and no chart,
styling, routing or UI library — the chart is inline SVG, the FFT is ~60 lines, the router is
~20. That is deliberate: fewer things to pin, to audit and to break on a machine that is not
this one. Hash routing also means the build deploys to a subdirectory, or opens from
`file://` off a USB stick, with no server rewrite rules — which is the thing that breaks ten
minutes before a demo.

## `npm test` is not optional

Writing those gates caught **two real bugs in the image demo**, both of which produced a
plausible-looking picture and a wrong number:

1. A non-conjugate-symmetric mask makes `P_N x` complex for a real image, so taking the real
   part for display silently leaves the null space. Data-consistency residual **4.7e-2**
   instead of machine precision — the demo's central claim, quietly false.
2. The fully-sampled "centre" was placed at index `n/2`. In natural FFT ordering DC is index
   **0**, so that sampled the *highest* frequencies fully and omitted DC: backwards
   physically, and not symmetric under `i -> n - i`. Residual **2.0e-5**.

Neither was visible by looking at the images. The suite includes a **control** — an
unprojected blob, which must be detectable at ~1e-2 — so the main assertion cannot pass
vacuously.

## Design and motion

Dark and light, both driven by tokens on `:root` with a `data-theme` override, so the toggle
and the OS setting agree. Inter and IBM Plex Mono from Google Fonts; everything else is
hand-written CSS — no framework.

**Every animation carries information rather than decorating.** The hero is the real
`randomMask1d` from the demo, looping from R=2 to R=12: lit rows are measured phase-encode
lines, dark rows are the null space, and the caption counts them, so the thing the project is
about is the first thing on the page. The coverage bars grow *from the axis* with
`transform-origin` pinned there, so the eye tracks distance from the 0.90 rule — the quantity
being claimed. The headline 69.3% counts up from 90%, which is the fall it describes.

Reveal-on-scroll uses one `IntersectionObserver` hook that fires once and disconnects;
content that re-animates on every pass is distracting rather than polished. Route changes
remount `main` via `key`, which replays the entry animation and resets demo state — a stale
canvas from the previous page is worse than a reset.

**`prefers-reduced-motion` is honoured throughout**, and not by hiding content: the reveal
hook starts in the shown state and the count-up jumps to its final value, so nothing is ever
stuck behind an animation that will not play.

## Where the numbers come from

**`src/data.js` is the only place a number appears.** Each block names the document it came
from (`T2.7`, `T2.8`, the demo config). Components read from it and format it; none of them
contain a figure of their own. If a result changes in the repository it changes in one file
here, which is the same rule the analysis scripts follow for label thresholds — one
definition, so two copies cannot drift.

**Every measured figure is development output.** The banks were built with a natural-image
prior applied to medical anatomy (D-016), so magnitudes will move when they are re-run under
the medical priors now wired. The *structure* is what the site shows: monotone coverage loss
with operator shift, restoration at every operator, and the distance ordering. The footer
says so, and the demo section says so again where it matters.

## Regenerating the demo table

The numbers in the "Live demo" section are the output of:

```bash
# from Project/
PYTHONPATH=src python3 -m caliber.runner configs/experiments/demo_coverage_failure.yaml
PYTHONPATH=src python3 scripts/analyse_bank.py runs/<run_id> \
    --calibration-operator mri.random1d.R2.c0.08.s0 --label ratio --tau 1.10
```

Both run on synthetic phantoms with a reference sampler, so they need no dataset, no
checkpoint and no GPU — and the harness stamps the result `NOT REPORTABLE` on its own. If
you change the config, update `DEMO` in `src/data.js` to match rather than leaving the two
out of step.

## Deploying

`npm run build` writes a fully static `dist/`. `vite.config.js` sets `base: './'`, so the
asset URLs are relative and the build works from a subdirectory — a GitHub Pages project
site, or a university server path — with no further configuration.

- **AWS Amplify Hosting:** `amplify.yml` at the **repository root** already has it. Connect
  the repo, and in the console enable **"Connect a monorepo"** with app root `site` — without
  that setting `appRoot` is ignored, Amplify builds the repo root, finds no `package.json`
  and fails. Two details in that spec are load-bearing:
  - `nvm use 22`. Vite 8 declares `engines: node ^20.19.0 || >=22.12.0`, and Amplify's
    default image has shipped older Node. The failure is an opaque resolution error, not a
    clear version message.
  - **No SPA rewrite rule is needed.** Hash routing means every requested URL is
    `/index.html` and the fragment never reaches the server, so the usual "404 → /index.html
    200" rule is unnecessary. It becomes mandatory only if routing is switched to history
    mode.
- **GitHub Pages / Netlify / Vercel:** build command `npm run build`, publish directory
  `dist`, project root `site`.
- **Anywhere else:** copy `dist/` to the web root. There is no server-side component.

## Layout

```
site/
  index.html
  vite.config.js
  src/
    main.jsx                     entry
    App.jsx                      the page, section by section
    data.js                      every number, with its source document
    index.css                    design tokens, light and dark
    components/
      CoverageChart.jsx          inline SVG, fixed 0.6-1.0 window
```

The chart's y-window is **fixed at 0.6–1.0 on purpose**. The claim is the distance from the
0.90 nominal line, and an axis that auto-scaled to the data would either flatter or
exaggerate it depending on which operators happened to be included.
