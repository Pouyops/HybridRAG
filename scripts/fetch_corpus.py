"""Download the HTTP core RFCs and convert them to Markdown for indexing.

The corpus is the IETF "HTTP core" set of Internet Standards (STD 97/98/99):

    RFC 9110  HTTP Semantics
    RFC 9111  HTTP Caching
    RFC 9112  HTTP/1.1

Together they are ~100k words of dense, normative, heavily cross-referenced
text: caching rules depend on method and status-code definitions in RFC 9110,
message framing in RFC 9112 depends on content codings defined in RFC 9110,
and so on. That makes them a realistic target for a "standards assistant"
RAG system, where answers must be grounded in a specific section and many
questions span more than one section or document.

The HTML comes from the HTTP Working Group's own rendering
(github.com/httpwg/httpwg.github.io). Conversion keeps the full text of each
RFC unchanged and only changes its format: numbered section titles become
Markdown headings (so Markdown-header chunking has real sections to split
on), paragraphs and list items become plain text, and ABNF/code blocks
become fenced code blocks. The navigation sidebar, table of contents and
alphabetical index are dropped since they are rendering artifacts, not text.

RFCs are published by the IETF Trust; see the "Copyright Notice" section
kept at the top of each converted file.

Usage:
    python scripts/fetch_corpus.py                # writes ./data/rfc91xx-*.md
    python scripts/fetch_corpus.py --out-dir DIR
"""

import argparse
import os
import re
import urllib.request

from bs4 import BeautifulSoup, NavigableString, Tag

SOURCE_URL = "https://raw.githubusercontent.com/httpwg/httpwg.github.io/main/specs/rfc{number}.html"

RFCS = {
    "9110": "rfc9110-http-semantics.md",
    "9111": "rfc9111-http-caching.md",
    "9112": "rfc9112-http-1.1.md",
}

# Rendering artifacts with no standalone meaning once flattened to text.
_SKIPPED_SECTION_IDS = {"rfc.index"}
_SKIPPED_CLASSES = {"toc", "sidebar", "navbar"}
_HEADING_LEVEL = {"h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def _inline(el) -> str:
    text = el.get_text() if isinstance(el, Tag) else str(el)
    return re.sub(r"\s+", " ", text).strip()


def _list_lines(list_el: Tag, depth: int = 0) -> list:
    """Render a <ul>/<ol> as "- item" lines, indenting nested lists rather
    than flattening them into their parent item's text."""
    lines = []
    for li in list_el.find_all("li", recursive=False):
        nested = [c for c in li.find_all(["ul", "ol"]) if c.find_parent("li") is li]
        for sub in nested:
            sub.extract()
        text = _inline(li)
        if text:
            lines.append("  " * depth + "- " + text)
        for sub in nested:
            lines.extend(_list_lines(sub, depth + 1))
    return lines


def _separate_blocks(soup: BeautifulSoup) -> None:
    """Text from adjacent block elements (e.g. <p>s inside one <li>) would
    otherwise run together when flattened ("...7);an Expires"); add a space
    after each one. Whitespace is collapsed later, so extra spaces are free."""
    for tag in soup.find_all(["p", "div", "li", "dt", "dd", "td", "th", "br"]):
        if tag.find_parent("pre") is None:
            tag.insert_after(" ")


def _is_skipped(el: Tag) -> bool:
    if el.name in ("script", "style", "nav", "hr"):
        return True
    if el.get("id") in _SKIPPED_SECTION_IDS:
        return True
    return bool(_SKIPPED_CLASSES.intersection(el.get("class") or []))


def _render(el: Tag, out: list) -> None:
    """Append Markdown blocks for `el`'s children to `out`, in document order."""
    for child in el.children:
        if isinstance(child, NavigableString):
            text = _inline(child)
            if text:
                out.append(text)
            continue
        if not isinstance(child, Tag) or _is_skipped(child):
            continue

        name = child.name
        if name in _HEADING_LEVEL:
            out.append("#" * _HEADING_LEVEL[name] + " " + _inline(child))
        elif name == "p":
            text = _inline(child)
            if text:
                out.append(text)
        elif name in ("ul", "ol"):
            out.append("\n".join(_list_lines(child)))
        elif name == "dl":
            lines = []
            for item in child.find_all(["dt", "dd"], recursive=False):
                text = _inline(item)
                lines.append(f"- **{text}**" if item.name == "dt" else f"  {text}")
            out.append("\n".join(lines))
        elif name == "pre":
            out.append("```\n" + child.get_text().strip("\n") + "\n```")
        elif name == "table":
            rows = []
            caption = child.find("caption")
            if caption:
                rows.append(_inline(caption))
            for tr in child.find_all("tr"):
                cells = [_inline(cell) for cell in tr.find_all(["th", "td"])]
                if any(cells):
                    rows.append(" | ".join(cells))
            out.append("\n".join(rows))
        elif name == "aside":
            text = _inline(child)
            if text:
                out.append("> " + text)
        else:
            # Structural wrappers (section, div, header, figure, ...).
            _render(child, out)


def html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    _separate_blocks(soup)
    title = _inline(soup.title)  # e.g. "RFC 9111 - HTTP Caching"
    main = soup.find("div", class_="main")
    if main is None:
        raise ValueError("Unexpected HTML layout: no <div class='main'> found.")

    blocks = [f"# {title}"]
    _render(main, blocks)
    text = "\n\n".join(block for block in blocks if block.strip())
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def fetch(number: str) -> str:
    with urllib.request.urlopen(SOURCE_URL.format(number=number), timeout=60) as resp:
        return resp.read().decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", default="./data/", help="Where to write the .md files.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for number, filename in RFCS.items():
        markdown = html_to_markdown(fetch(number))
        path = os.path.join(args.out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"RFC {number}: {len(markdown.split()):,} words -> {path}")


if __name__ == "__main__":
    main()
