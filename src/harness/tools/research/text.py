"""HTML -> readable text, and extraction of the passages relevant to a query.

Pages are never passed to the model whole: they are split into sections and only the
best-matching sections (plus code blocks near them) are kept, within a character budget.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

from harness.tools.base import TRUNCATION_MARKER

_SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside"}
_BLOCK_TAGS = {"p", "div", "section", "article", "li", "tr", "br", "dd", "dt", "table", "ul", "ol"}
_HEADINGS = {"h1": "#", "h2": "##", "h3": "###", "h4": "####", "h5": "#####", "h6": "######"}
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{1,}")
_STOP = {
    "the", "and", "for", "with", "how", "what", "use", "using", "into", "from", "that", "this",
    "does", "can", "are", "is", "to", "in", "of", "a", "an", "on", "it", "be", "or", "do",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False
        self._in_pre = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _HEADINGS:
            self.parts.append(f"\n\n{_HEADINGS[tag]} ")
        elif tag == "pre":
            self._in_pre += 1
            self.parts.append("\n```\n")
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag == "pre" and self._in_pre:
            self._in_pre -= 1
            self.parts.append("\n```\n")
        elif tag in _HEADINGS or tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data.strip()
        elif not self._skip:
            self.parts.append(data if self._in_pre else re.sub(r"\s+", " ", data))


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, text) with headings as '#' lines and <pre> blocks as code fences."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return parser.title, text.strip()


def query_terms(query: str) -> set[str]:
    terms: set[str] = set()
    for word in _WORD.findall(query):
        for part in [word, *word.split(".")]:  # "statistics.mean" also matches "mean"
            if len(part) > 1 and part.lower() not in _STOP:
                terms.add(part.lower())
    return terms


def _window(body: str, terms: set[str], size: int) -> str:
    """Cut ``body`` to ``size`` chars, centred on the first query-term hit, marking cuts."""
    lowered = body.lower()
    hits = [i for i in (lowered.find(t) for t in terms) if i >= 0]
    start = max(min(hits) - size // 4, 0) if hits else 0
    end = min(start + size, len(body))
    start = max(end - size, 0)
    prefix = f"{TRUNCATION_MARKER}: earlier text cut]\n" if start > 0 else ""
    suffix = f"\n{TRUNCATION_MARKER}: section cut]" if end < len(body) else ""
    return prefix + body[start:end] + suffix


@dataclass(frozen=True)
class Excerpt:
    heading: str
    text: str
    score: float


def _sections(text: str) -> list[tuple[str, str]]:
    """Split on markdown-style headings; long sections are split into paragraphs groups."""
    sections: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            if buffer:
                sections.append((heading, "\n".join(buffer).strip()))
            heading, buffer = line.lstrip("# ").strip(), []
        else:
            buffer.append(line)
    if buffer:
        sections.append((heading, "\n".join(buffer).strip()))
    out: list[tuple[str, str]] = []
    for head, body in sections:
        if len(body) <= 1_500:
            if body:
                out.append((head, body))
            continue
        chunk: list[str] = []
        size = 0
        for para in re.split(r"\n\s*\n", body):
            if size + len(para) > 1_500 and chunk:
                out.append((head, "\n\n".join(chunk)))
                chunk, size = [], 0
            chunk.append(para)
            size += len(para)
        if chunk:
            out.append((head, "\n\n".join(chunk)))
    return out


def extract_relevant(
    text: str, query: str | None, *, max_chars: int, max_excerpts: int = 6
) -> tuple[list[Excerpt], bool]:
    """Pick the sections most relevant to ``query`` within ``max_chars``.

    Returns (excerpts in document order, truncated?). Without a query, the leading sections
    are returned. Truncation is always marked with [OUTPUT TRUNCATED].
    """
    sections = _sections(text)
    if not sections:
        return [], False
    terms = query_terms(query or "")

    def score(index: int, heading: str, body: str) -> float:
        if not terms:
            return 1.0 / (index + 1)
        haystack = (heading + " " + body).lower()
        hits = sum(haystack.count(t) for t in terms)
        coverage = sum(1 for t in terms if t in haystack) / len(terms)
        bonus = 0.5 if "```" in body and hits else 0.0
        heading_bonus = sum(1 for t in terms if t in heading.lower())
        return coverage * 3 + min(hits, 20) * 0.1 + bonus + heading_bonus

    scored = sorted(
        ((score(i, h, b), i, h, b) for i, (h, b) in enumerate(sections)),
        key=lambda item: (-item[0], item[1]),
    )
    chosen: list[tuple[int, Excerpt]] = []
    used = 0
    truncated = len(sections) > max_excerpts
    for s, index, heading, body in scored:
        if len(chosen) >= max_excerpts or (terms and s <= 0):
            truncated = True
            break
        remaining = max_chars - used
        if remaining < 200:
            truncated = True
            break
        if len(body) > remaining:
            body = _window(body, terms, remaining - 90)
            truncated = True
        chosen.append((index, Excerpt(heading, body, round(s, 3))))
        used += len(body) + len(heading)
    return [e for _, e in sorted(chosen, key=lambda item: item[0])], truncated
