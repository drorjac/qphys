"""The documentation must not rot.

`docs/` is organised one folder per sub-project, and the topic pages carry
the figures. Both are easy to break silently by moving a file, so they are
checked rather than trusted.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOPICS = ("collapse", "lawlearn")
SKIP = ("http", "https", "mailto:", "#")


def _markdown_files():
    for path in sorted(ROOT.rglob("*.md")):
        if any(p in {".venv", "node_modules", ".pytest_cache"} for p in path.parts):
            continue
        yield path


def _links(text: str):
    return re.findall(r"\]\(([^)#][^)]*)\)", text) + re.findall(r'src="([^"]+)"', text)


def test_every_relative_link_resolves():
    broken = []
    for path in _markdown_files():
        for target in _links(path.read_text()):
            if target.startswith(SKIP) or "<" in target:
                continue
            if not (path.parent / target).resolve().exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
    assert not broken, "broken links:\n  " + "\n  ".join(broken)


@pytest.mark.parametrize("topic", TOPICS)
def test_every_topic_has_a_docs_folder(topic):
    page = ROOT / "docs" / topic / "README.md"
    assert page.exists(), f"docs/{topic}/README.md is missing"
    assert len(page.read_text().split()) > 150, f"docs/{topic} is a stub"


@pytest.mark.parametrize("topic", TOPICS)
def test_every_topic_page_shows_figures(topic):
    """A topic page that shows no pictures is a table of contents."""
    text = (ROOT / "docs" / topic / "README.md").read_text()
    images = [t for t in _links(text) if t.endswith((".png", ".gif"))]
    assert len(images) >= 3, f"docs/{topic} embeds only {len(images)} figures"


@pytest.mark.parametrize("topic", TOPICS)
def test_each_topic_matches_a_package(topic):
    """The docs layout follows the code layout, not a separate taxonomy."""
    assert (ROOT / "src" / "qphys" / topic).is_dir()


def test_the_docs_index_links_every_topic():
    index = (ROOT / "docs" / "README.md").read_text()
    for topic in TOPICS:
        assert f"({topic}/)" in index, f"docs/README.md does not link {topic}"


def test_the_readme_sends_readers_to_the_topic_folders():
    assert "docs/collapse" in (ROOT / "README.md").read_text()
