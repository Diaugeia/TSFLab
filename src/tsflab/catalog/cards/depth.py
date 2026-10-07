"""Progressive-disclosure views of model, component, and dataset cards.

One depth model serves every catalog resource, so an agent can spend context
only as needed:

* L0 - one line: ``name``, ``kind``, the README ``description``, ``tags``.
* L1 - runtime facts derived from code, ``card.toml`` facts, and the README body.
* L2 - L1 plus ``reference.md`` when the card has one.
* L3 - the source, config, and card paths to open next.

Cards follow ``tsflab.card/1`` (see ``tsflab.catalog.cards.schema``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re


DEPTHS = (0, 1, 2, 3)
DEPTH_HELP = (
    "0 one-line summary, 1 front matter + interface/constraints, "
    "2 full card, 3 paths to open"
)
L0_SUMMARY_CHARS = 160
# Sections shown at L1: how the resource is called and what limits composition.


@dataclass(frozen=True)
class Card:
    """A parsed card: flat front matter plus ordered level-two sections."""

    path: Path
    front: dict[str, object]
    text: str
    sections: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    #: ``card.toml`` verbatim, the README body (L1), and ``reference.md`` (L2).
    facts_text: str = ""
    body: str = ""
    reference: str | None = None

    @property
    def name(self) -> str:
        return str(self.front.get("name", self.path.parent.name))

    @property
    def kind(self) -> str:
        return str(self.front.get("kind") or "model")

    @property
    def summary(self) -> str:
        return " ".join(str(self.front.get("summary", "")).split())

    @property
    def headline(self) -> str:
        """The L0 text: the README description (Skill-style what + when to use)."""
        return " ".join(str(self.front.get("description") or self.front.get("tagline") or self.summary).split())

    @property
    def tags(self) -> tuple[str, ...]:
        return card_tags(self.front)


def split_sections(text: str) -> tuple[tuple[str, str], ...]:
    """Split a card body on level-two headings, ignoring fenced code blocks."""
    sections: list[tuple[str, list[str]]] = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if not fenced and re.fullmatch(r"<!-- [\w:-]+ -->", line.strip()):
            continue  # generated-block markers are not content
        if not fenced and line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif sections:
            sections[-1][1].append(line)
    return tuple((title, "\n".join(body).strip()) for title, body in sections)


def read_card(path: Path) -> Card:
    """Parse one card directory (or its README); raise ``ValueError`` when malformed."""
    from tsflab.catalog.cards.store import load

    directory = path if path.is_dir() else path.parent
    files = load(directory)
    front = {**files.facts, "summary": files.description, "description": files.description}
    readme = directory / "README.md"
    text = readme.read_text(encoding="utf-8")
    if files.reference:
        text = text.rstrip() + "\n\n" + files.reference
    return Card(
        path=readme,
        front=front,
        text=text,
        sections=files.sections + files.reference_sections,
        facts_text=(directory / "card.toml").read_text(encoding="utf-8"),
        body=files.body,
        reference=files.reference,
    )


def card_tags(front: dict[str, object]) -> tuple[str, ...]:
    """Return explicit tags, else derive them from list fields and the loader."""
    tags = front.get("tags")
    if isinstance(tags, list):
        return tuple(str(tag) for tag in tags)
    derived: list[str] = []
    for key, value in front.items():
        if key in {"tags", "origin_models"}:
            continue
        if isinstance(value, list):
            derived.extend(str(item) for item in value)
    for key in ("category", "loader"):
        if front.get(key):
            derived.append(str(front[key]))
    return tuple(dict.fromkeys(derived))


def truncate_summary(summary: str, limit: int = L0_SUMMARY_CHARS) -> str:
    """Return the first sentence, shortened to ``limit`` characters."""
    summary = " ".join(summary.split())
    first = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`])", summary, maxsplit=1)[0]
    if len(first) <= limit:
        return first
    return first[: limit - 3].rstrip(" ,;:") + "..."


def l0_record(
    name: str, kind: str, summary: str, tags: tuple[str, ...] | list[str]
) -> dict[str, object]:
    """Return the structured L0 record for one resource."""
    return {
        "name": name,
        "kind": kind,
        "summary": truncate_summary(summary),
        "tags": list(tags),
    }


def l0_line(record: dict[str, object]) -> str:
    """Format an L0 record as one tab-separated line: name, kind, summary, tags."""
    tags = ",".join(str(tag) for tag in record["tags"])  # type: ignore[union-attr]
    return f"{record['name']}\t{record['kind']}\t{record['summary']}\t{tags}"


def card_l0(card: Card) -> dict[str, object]:
    return l0_record(card.name, card.kind, card.headline, card.tags)


def approx_tokens(text: str) -> int:
    """Rough token count (characters / 4) used to price the next depth."""
    return max(1, len(text) // 4)


def front_matter_text(card: Card) -> str:
    """Return the card's facts (``card.toml``) verbatim."""
    return card.facts_text.rstrip()


def l1_sections(card: Card) -> tuple[tuple[str, str], ...]:
    """Return the README (L1) sections."""
    from tsflab.catalog.cards.store import split_sections as _split

    return _split(card.body)


def render_text(
    card: Card,
    depth: int,
    *,
    facts: dict[str, object] | None = None,
    paths: list[str] | None = None,
) -> str:
    """Render ``card`` at ``depth`` as text.

    ``facts`` are runtime facts (for example config paths or dataset split
    borders) that live outside the card; they are listed before the front
    matter at L1.
    ``paths`` are the files to open at L3.
    """
    line = l0_line(card_l0(card))
    if depth == 0:
        return line
    if depth == 3:
        return "\n".join([line, "", *(f"- {path}" for path in (paths or []))])
    parts = [line]
    if facts:  # generated from code, so they lead the page
        parts.append(
            "Runtime facts:\n"
            + "\n".join(f"- {key}: {_format_fact(value)}" for key, value in facts.items())
        )
    parts.append(front_matter_text(card))
    parts.append(card.body.rstrip())
    if depth == 2 and card.reference:
        parts.append(card.reference.rstrip())
    elif depth == 1 and card.reference:
        from tsflab.catalog.cards.store import split_sections as _split

        titles = ", ".join(title for title, _ in _split(card.reference))
        parts.append(f"Next: --depth 2 adds reference.md ({titles}; ~{approx_tokens(card.reference)} tokens); "
                     "--depth 3 lists the files to open.")
    elif depth == 1:
        parts.append("Next: --depth 3 lists the files to open (this card has no reference.md).")
    return "\n\n".join(parts) + "\n"


def _format_fact(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "(none)"
    return str(value)


def card_payload(
    card: Card,
    depth: int,
    *,
    facts: dict[str, object] | None = None,
    paths: list[str] | None = None,
) -> dict[str, object]:
    """Return the structured form of ``render_text`` for ``--json`` output."""
    payload: dict[str, object] = {"depth": depth, **card_l0(card)}
    if depth in (1, 2):
        payload["card"] = card.front
        payload["facts"] = facts or {}
        payload["sections"] = dict(l1_sections(card))
    if depth == 2:
        payload["reference"] = card.reference or ""
    if depth == 3:
        payload["paths"] = paths or []
    return payload
