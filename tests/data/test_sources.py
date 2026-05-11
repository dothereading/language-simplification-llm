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


class TestRandomSummaryRobustness:
    """A single bad random article (400 from a malformed redirect, etc.) must
    not kill the iterator — it should return None so the caller skips and
    fetches another."""

    def _fake_session(self, status_code: int):
        class FakeResp:
            def __init__(self, code):
                self.status_code = code
                self.headers = {}
            def json(self):
                return {}
            def raise_for_status(self):
                if self.status_code >= 400:
                    import requests
                    raise requests.HTTPError(f"{self.status_code} error")

        class FakeSession:
            def get(self, *a, **kw):
                return FakeResp(status_code)
        return FakeSession()

    def test_400_returns_none_not_raises(self):
        # Real failure mode: Wikipedia redirected /random/summary to a
        # /summary/<title-with-slash> URL that 400'd.
        sess = self._fake_session(400)
        assert mod._random_summary(sess, max_retries=2) is None

    def test_404_returns_none(self):
        sess = self._fake_session(404)
        assert mod._random_summary(sess, max_retries=2) is None


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
