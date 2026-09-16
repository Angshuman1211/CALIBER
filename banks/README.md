# Shared score banks

Score banks committed deliberately so another track can work from a copied directory —
which is what `score_bank_spec.md` §3 says a bank is *for*: "a bank is self-describing,
so Track 3 can compute operator distances and Track 1 can calibrate from a copied
directory, with no access to the machine that produced it."

Everything under `runs/` stays local and gitignored. What lands here is the exception,
and the exception has rules.

## What may be committed here

**Only banks from open substrates** — phantom, M4Raw — and only when another track needs
one. A score bank is a table of per-region scalars, not images, and M4Raw is CC-BY, so
redistributing derived numbers from it is permitted with attribution.

**fastMRI-derived banks must not be committed here.** The NYU Data Sharing Agreement
forbids redistribution, and whether that reaches *derived* score banks is an open question
this project has not answered — `T2.1_data_and_operator_pipeline.md` §7 flags it for
confirmation before the Week 12 open-source release (O7). Until it is answered, the
conservative reading applies: the Week 3 fastMRI bank stays on the machine that produced
it. Ask, and copy it directly, rather than committing it here.

**Never reconstructions.** `recons/` is excluded from everything here. Cached posterior
samples are images, they are orders of magnitude larger than the score table, and the
release rule is code, configs and score banks — never images.

## What is here

### `w02_solver_smoke/` — 384 rows

The Week 2 solver smoke run, and the project's **first bank produced by a real pretrained
prior**: DDNM over `ddpm-celebahq-256` on real M4Raw k-space, two operators (R=4, R=8),
24 steps, 4 draws, 16 px regions. Requested by Track 1 (D-038 ask (a)) as the only
real-prior output in the project.

**It is not reportable, and its manifest says so** — `provenance.reportable: false`. The
prior is a face model (D-016) and M4Raw is a development substrate (D-015). It exercises
the pipeline; it is not evidence about anatomy.

Two things about its age, both harmless and both worth knowing:

- It predates `s_range` (D-038 ask (b)) and carries no such column. That column is
  nullable, so the bank still validates as it is; `coerce` adds it as NA if you want the
  full schema. (`read_bank` validates but does not backfill — verified, not assumed.)
- It predates `complete` in the manifest (D-040). A manifest without that key is read as
  complete, which this one is.

```python
from caliber.scorebank import read_bank, read_manifest
df = read_bank("banks/w02_solver_smoke")        # 384 rows, 16 px regions
read_manifest("banks/w02_solver_smoke")["provenance"]["reportable"]   # False
```

**Attribution.** Derived from M4Raw (Lyu et al., CC-BY), which requires attribution on
derived works. Reconstructions come from `google/ddpm-celebahq-256` (Apache-2.0).
