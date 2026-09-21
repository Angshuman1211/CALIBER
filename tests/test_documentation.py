"""The Track 2 decision index stays in step with the cross-track log.

`T2_problems_and_decisions.md` Part II indexes every entry of `decision_log.md`, so a
problem can be traced to the decision that settled it. It fell six rows behind within a
fortnight: two of Track 1's entries arrived in a merge and never got rows, and so did one of
Track 2's own. Nothing noticed, because nothing checked. Now something does.
"""

from __future__ import annotations

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "Documentation"


def _log_ids() -> list[str]:
    text = (DOCS / "decision_log.md").read_text(encoding="utf-8")
    return re.findall(r"^## (D-\d{3}) · ", text, re.M)


def _index_ids() -> list[str]:
    text = (DOCS / "T2_problems_and_decisions.md").read_text(encoding="utf-8")
    return re.findall(r"^\| \[(D-\d{3})\]\(decision_log\.md\)", text, re.M)


def test_the_log_has_no_duplicate_decision_numbers():
    """Two tracks writing D-0NN in parallel collided twice; the log must stay unique."""
    ids = _log_ids()
    assert len(ids) == len(set(ids)), sorted({i for i in ids if ids.count(i) > 1})


def test_every_decision_in_the_log_has_exactly_one_index_row():
    log, index = _log_ids(), _index_ids()
    assert not set(log) - set(index), f"missing from the index: {sorted(set(log) - set(index))}"
    assert not set(index) - set(log), f"indexed but not in the log: {sorted(set(index) - set(log))}"
    assert len(index) == len(set(index)), "an index row is duplicated"


def test_the_index_is_in_numeric_order():
    nums = [int(i[2:]) for i in _index_ids()]
    assert nums == sorted(nums)


def test_the_index_states_its_own_size_correctly():
    text = (DOCS / "T2_problems_and_decisions.md").read_text(encoding="utf-8")
    stated = int(re.search(r"the index — (\d+) entries", text).group(1))
    assert stated == len(_index_ids())
