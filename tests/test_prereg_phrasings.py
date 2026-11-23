"""The documentation must not use the phrasings the pre-registration forbids.

Three claims in this project are measured and false, or measured and overstated, and each
has drifted back into a document at least once after being corrected -- which is why they
live in :data:`caliber.evaluation.prereg.FORBIDDEN_PHRASINGS` with the replacement wording
attached, and why this test exists rather than a note asking people to remember.

Two details earned by getting this wrong once. **Whitespace is normalised before searching**,
because Markdown wraps prose and the first version of this test missed a phrase that was
split across two lines -- which means it would also have missed a bare *use* split the same
way. And a phrase is allowed when the surrounding text *prohibits* it, since that is how the
rule itself has to be written down; the check is a negation marker inside a window around the
occurrence rather than on the same line, for the same wrapping reason. The heuristic is
deliberately loose: a false pass costs one review comment, a false failure costs a test
nobody can get green.
"""

from __future__ import annotations

import re
from pathlib import Path

from caliber.evaluation import prereg
from caliber.evaluation.prereg import FORBIDDEN_PHRASINGS

DOCS = Path(__file__).resolve().parents[1] / "Documentation"

#: Text within this many characters either side of an occurrence counts as its context.
WINDOW = 160

#: Context carrying one of these is talking *about* the phrase, not using it.
PROHIBITION = re.compile(
    r"never|not\b|n't|forbidden|instead|write |avoid|rather than|must", re.IGNORECASE
)


def _normalised(path: Path) -> str:
    """Document text with every run of whitespace collapsed to one space."""
    return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))


def _occurrences(text: str, phrase: str):
    """Every position of ``phrase``, case-insensitively, with its surrounding context."""
    low, needle = text.lower(), phrase.lower()
    start = low.find(needle)
    while start != -1:
        yield start, text[max(0, start - WINDOW) : start + len(needle) + WINDOW]
        start = low.find(needle, start + 1)


def test_every_forbidden_phrasing_carries_its_replacement():
    """The rule is useless without the wording to use in its place."""
    for bad, guidance in FORBIDDEN_PHRASINGS.items():
        assert len(guidance) > 40, f"{bad!r} has no usable replacement guidance"


def test_no_document_uses_a_forbidden_phrasing_bare():
    offences = []
    for md in sorted(DOCS.glob("*.md")):
        text = _normalised(md)
        for bad in FORBIDDEN_PHRASINGS:
            for pos, context in _occurrences(text, bad):
                if not PROHIBITION.search(context):
                    offences.append(f"{md.name} at char {pos}: {bad!r} in ...{context}...")
    assert not offences, "\n".join(
        ["forbidden phrasings used without prohibition:", *offences]
    )


def test_the_prohibitions_themselves_are_still_written_down():
    """Guard against the rule being deleted along with the phrase it governs.

    If no document mentions a forbidden phrasing at all, the project has lost the record of
    why it is forbidden, and the next person to write it has nothing to find.
    """
    corpus = " ".join(_normalised(md) for md in sorted(DOCS.glob("*.md"))).lower()
    missing = [bad for bad in FORBIDDEN_PHRASINGS if bad.lower() not in corpus]
    assert not missing, f"no document records why these are forbidden: {missing}"


def test_the_scan_would_catch_a_wrapped_bare_use(tmp_path):
    """The regression the rewrite was for: a bare use split across two lines."""
    doc = tmp_path / "draft.md"
    doc.write_text(
        "Our experiments show that the curves\ncollapse onto a shared trend.\n",
        encoding="utf-8",
    )
    text = _normalised(doc)
    hits = [c for _, c in _occurrences(text, "the curves collapse")]
    assert hits, "whitespace normalisation must join the wrapped phrase"
    assert not PROHIBITION.search(hits[0]), "this context is a bare use, not a prohibition"


def test_unsafe_label_source_is_frozen_with_its_sensitivity():
    """D86: the label's source split is frozen, and the direction it cuts is reported.

    The across-split label removes a bias that *inflates* the false-safe rate, so the
    amendment lowers Track 1's own primary endpoint. A change in that direction is only
    legitimate if the other arm stays visible, which is what the sensitivity tuple is for.
    """
    assert prereg.UNSAFE_LABEL_SOURCE == "across_split_mean"
    assert prereg.UNSAFE_LABEL_MIN_SPLITS >= 8
    assert "registry_split" in prereg.UNSAFE_LABEL_SOURCE_SENSITIVITY
    assert prereg.UNSAFE_LABEL_SOURCE in prereg.UNSAFE_LABEL_SOURCE_SENSITIVITY
    # the freeze travels into every result's provenance block, or it is not a freeze
    assert prereg.as_dict()["unsafe_label_source"] == prereg.UNSAFE_LABEL_SOURCE


def test_amendments_record_the_endpoint_lowering_change():
    """An amendment that helps the endpoint has to say so where the audit trail is."""
    text = " ".join(note for _, note in prereg.AMENDMENTS)
    assert "UNSAFE_LABEL_SOURCE" in text
    assert "LOWERS THE PRIMARY ENDPOINT" in text
    assert prereg.AMENDMENTS[-1][0] >= prereg.FROZEN_ON
