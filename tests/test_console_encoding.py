"""Printed strings must survive a Windows console (Track 2, Week 5).

`scripts/select_weights.py` crashed with `UnicodeEncodeError` on Track 2's machine printing
a `<=`. The project is developed on macOS, where `sys.stdout` is UTF-8, and run on Windows,
where a legacy console is cp1252 and raises on any character outside it. So the typography
that reads nicely in a docstring is a hard crash in a `print`, and the failure appears only
on the machine that produces every bank.

The split enforced here is deliberate rather than blanket:

* **docstrings keep their notation.** `||P_N x||`, `theta`, `alpha` and the rest are worth
  real characters in prose that explains the mathematics, and a docstring is never written
  to a console.
* **every other string literal is ASCII**, because any of them can reach `print`.

Markdown is unaffected — editors and GitHub are UTF-8 — so `Documentation/` keeps its
em-dashes. The rule is only about Python source.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted(
    set((REPO_ROOT / "src").rglob("*.py")) | set((REPO_ROOT / "scripts").rglob("*.py"))
)


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """Ids of the string constants that are docstrings, which are exempt."""
    out = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ) and ast.get_docstring(node, clean=False) is not None:
            out.add(id(node.body[0].value))
    return out


def printable_literals(path: Path) -> list[tuple[int, str]]:
    """Every non-docstring string literal in a file, with its line number."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    exempt = _docstring_nodes(tree)
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in exempt
    ]


def test_there_are_sources_to_check():
    """A path typo here would make every assertion below vacuously true."""
    assert len(SOURCES) > 20


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_printable_literal_contains_non_ascii(path: Path):
    offenders = [
        (line, sorted({c for c in value if ord(c) > 127}), value[:60])
        for line, value in printable_literals(path)
        if any(ord(c) > 127 for c in value)
    ]
    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} has string literals a cp1252 console cannot "
        f"print: {offenders}. Docstrings may keep their notation; anything that can "
        "reach print() may not."
    )


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_every_printable_literal_encodes_to_cp1252(path: Path):
    """The direct form of the same check, against the encoding that actually failed.

    Kept alongside the ASCII assertion because they are not quite the same: cp1252 accepts
    a few characters above 127 (it would take an em-dash, in fact). ASCII is the stricter
    rule and the one worth holding, but this test names the real failure mode, so a future
    reader can see which constraint came from where.
    """
    for line, value in printable_literals(path):
        try:
            value.encode("cp1252")
        except UnicodeEncodeError as exc:  # pragma: no cover - the guard, not the path
            pytest.fail(
                f"{path.relative_to(REPO_ROOT)}:{line} cannot be printed on a cp1252 "
                f"console: {exc}. Literal starts {value[:60]!r}"
            )
