import importlib.util
import os

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "fetch_corpus.py")
_spec = importlib.util.spec_from_file_location("fetch_corpus", _SCRIPT)
fetch_corpus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch_corpus)

HTML = """
<html><head><title>RFC 9999 - Test Protocol</title></head><body>
<nav>navbar junk</nav>
<div class="main">
  <div class="toc">1. Intro 2. Rules</div>
  <section id="intro"><h2>1. Intro</h2>
    <div><p>A cache <em>MUST NOT</em> store a response unless:</p></div>
    <ul>
      <li>the method is understood;</li>
      <li><p>the response contains one of:</p>
        <ul><li>a max-age directive;</li><li>an Expires field.</li></ul>
      </li>
    </ul>
    <section id="rules"><h3>1.1. Rules</h3>
      <pre>  rule = 1*DIGIT</pre>
      <dl><dt>Strong:</dt><dd>exact match</dd></dl>
    </section>
  </section>
  <section id="rfc.index"><h2>Index</h2><p>A B C</p></section>
</div>
</body></html>
"""


def test_html_to_markdown_keeps_structure_and_drops_navigation():
    markdown = fetch_corpus.html_to_markdown(HTML)

    assert markdown.startswith("# RFC 9999 - Test Protocol\n")
    assert "## 1. Intro" in markdown
    assert "### 1.1. Rules" in markdown
    assert "A cache MUST NOT store a response unless:" in markdown
    # Nested list items stay separate instead of running together.
    assert "- the response contains one of:\n  - a max-age directive;\n  - an Expires field." in markdown
    assert "```\n  rule = 1*DIGIT\n```" in markdown
    assert "- **Strong:**\n  exact match" in markdown
    for dropped in ("navbar junk", "1. Intro 2. Rules", "Index", "A B C"):
        assert dropped not in markdown
