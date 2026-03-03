from pathlib import Path

from ragged.parsers import get_parser, SUPPORTED_EXTENSIONS
from ragged.parsers.markdown import MarkdownParser
from ragged.parsers.html import HtmlParser
from ragged.parsers.text import TextParser

FIXTURES = Path(__file__).parent / "fixtures"


class TestParserRegistry:
    def test_get_parser_returns_correct_type_for_md(self):
        parser = get_parser("docs/readme.md")
        assert isinstance(parser, MarkdownParser)

    def test_get_parser_returns_correct_type_for_html(self):
        parser = get_parser("page.html")
        assert isinstance(parser, HtmlParser)

    def test_get_parser_returns_correct_type_for_txt(self):
        parser = get_parser("notes.txt")
        assert isinstance(parser, TextParser)

    def test_get_parser_returns_none_for_unknown(self):
        parser = get_parser("image.png")
        assert parser is None

    def test_supported_extensions_includes_common_types(self):
        for ext in [".md", ".pdf", ".html", ".docx", ".txt"]:
            assert ext in SUPPORTED_EXTENSIONS


class TestMarkdownParser:
    def test_parses_markdown_with_frontmatter(self):
        docs = MarkdownParser().parse(FIXTURES / "sample.md")
        assert len(docs) == 1
        doc = docs[0]
        assert "Getting Started" in doc.text
        assert doc.metadata["section"] == "Sample Documentation"
        assert doc.metadata["source"] == str(FIXTURES / "sample.md")

    def test_parses_markdown_content(self):
        docs = MarkdownParser().parse(FIXTURES / "sample.md")
        assert "pip install ragged" in docs[0].text

    def test_metadata_has_required_keys(self):
        docs = MarkdownParser().parse(FIXTURES / "sample.md")
        meta = docs[0].metadata
        assert "source" in meta
        assert "page" in meta
        assert "section" in meta


class TestHtmlParser:
    def test_parses_html_extracts_text(self):
        docs = HtmlParser().parse(FIXTURES / "sample.html")
        assert len(docs) == 1
        assert "Hybrid retrieval" in docs[0].text

    def test_extracts_title(self):
        docs = HtmlParser().parse(FIXTURES / "sample.html")
        assert docs[0].metadata["section"] == "Sample HTML Document"

    def test_removes_script_and_style(self):
        docs = HtmlParser().parse(FIXTURES / "sample.html")
        assert "console.log" not in docs[0].text
        assert "font-family" not in docs[0].text


class TestTextParser:
    def test_parses_text_file(self):
        docs = TextParser().parse(FIXTURES / "sample.txt")
        assert len(docs) == 1
        assert "plain text document" in docs[0].text

    def test_metadata_source_is_set(self):
        docs = TextParser().parse(FIXTURES / "sample.txt")
        assert str(FIXTURES / "sample.txt") in docs[0].metadata["source"]

    def test_empty_file_returns_empty_list(self, tmp_path):
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        docs = TextParser().parse(empty_file)
        assert docs == []
