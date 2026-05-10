"""Tests for the Wikipedia source helpers.

These exercise pure logic (regex filters, paragraph splitting) so we don't
hit the network. Currently importing from `wiki_source`; after the rename
the import will switch to `sources`.
"""
from __future__ import annotations

import pytest


def _import_module():
    import langsimp.data.sources as sources
    return sources


mod = _import_module()


class TestSplitParagraphs:
    def test_splits_on_blank_lines(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird."
        out = mod._split_paragraphs(text)
        assert out == ["First paragraph.", "Second paragraph.", "Third."]

    def test_collapses_internal_whitespace(self):
        text = "Line one\n  with    spaces."
        out = mod._split_paragraphs(text)
        assert out == ["Line one with spaces."]

    def test_drops_empty_chunks(self):
        text = "\n\n\nOnly one.\n\n\n"
        out = mod._split_paragraphs(text)
        assert out == ["Only one."]


class TestBadTitleRegex:
    @pytest.mark.parametrize("title", [
        "Foo (disambiguation)",
        "List of cities in France",
        "DISAMBIGUATION page",
    ])
    def test_matches_bad_titles(self, title):
        assert mod._BAD_TITLE_RE.search(title) is not None

    @pytest.mark.parametrize("title", [
        "Albert Einstein",
        "History of Rome",
        "Photosynthesis",
    ])
    def test_skips_normal_titles(self, title):
        assert mod._BAD_TITLE_RE.search(title) is None
