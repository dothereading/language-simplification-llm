"""Tests for the generate.py CLI helpers.

Model loading and generation are integration-only; the testable surface is
input parsing (text arg / --file / stdin) and paragraph splitting.
"""
from __future__ import annotations

import io
import pytest

generate = pytest.importorskip("langsimp.inference.generate", reason="generate.py not implemented yet (RED)")


class TestReadInput:
    def test_text_arg_returns_single_paragraph(self):
        out = generate.read_input(text="Hello world.", file_path=None, stdin=None)
        assert out == ["Hello world."]

    def test_file_path_reads_and_splits(self, tmp_path):
        f = tmp_path / "in.txt"
        f.write_text("First paragraph.\n\nSecond paragraph.\n\nThird.")
        out = generate.read_input(text=None, file_path=f, stdin=None)
        assert out == ["First paragraph.", "Second paragraph.", "Third."]

    def test_stdin_when_no_other_input(self):
        stdin = io.StringIO("From stdin paragraph one.\n\nParagraph two.")
        out = generate.read_input(text=None, file_path=None, stdin=stdin)
        assert out == ["From stdin paragraph one.", "Paragraph two."]

    def test_text_takes_precedence_over_file(self, tmp_path):
        # If the user passes both, the inline text wins (less surprising).
        f = tmp_path / "in.txt"
        f.write_text("from file")
        out = generate.read_input(text="from arg", file_path=f, stdin=None)
        assert out == ["from arg"]

    def test_strips_whitespace_only_paragraphs(self, tmp_path):
        f = tmp_path / "in.txt"
        f.write_text("Real text.\n\n   \n\nMore text.")
        out = generate.read_input(text=None, file_path=f, stdin=None)
        assert out == ["Real text.", "More text."]

    def test_collapses_internal_whitespace_within_a_paragraph(self, tmp_path):
        f = tmp_path / "in.txt"
        f.write_text("Line one\n  with    spaces.\n\nNext.")
        out = generate.read_input(text=None, file_path=f, stdin=None)
        assert out == ["Line one with spaces.", "Next."]

    def test_no_input_raises(self):
        with pytest.raises(ValueError):
            generate.read_input(text=None, file_path=None, stdin=None)


class TestFormatOutput:
    def test_single_paragraph_just_text(self):
        out = generate.format_output(["First."], ["Simplified first."], show_source=False)
        assert "Simplified first." in out
        # No header noise for a single output
        assert "PARAGRAPH" not in out

    def test_multi_paragraph_separates_with_dividers(self):
        out = generate.format_output(
            ["First.", "Second."], ["Simple 1.", "Simple 2."], show_source=False,
        )
        assert "Simple 1." in out
        assert "Simple 2." in out
        # Some kind of separator between them
        assert out.count("Simple") == 2

    def test_show_source_includes_input_text(self):
        out = generate.format_output(
            ["The original."], ["The simple."], show_source=True,
        )
        assert "The original." in out
        assert "The simple." in out
