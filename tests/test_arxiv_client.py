import httpx
import pytest

from app.config import Settings
from app.services.arxiv_client import ArxivClient

SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2101.00001v2</id>
    <title>  A Great Paper About
      Retrieval Augmented Generation </title>
    <summary>We propose a new method for RAG.</summary>
    <published>2021-01-01T00:00:00Z</published>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2101.00001v2" rel="related" type="application/pdf"/>
  </entry>
</feed>
"""


def test_parse_feed_extracts_expected_fields():
    settings = Settings(anthropic_api_key="test")
    client = ArxivClient(settings, httpx.AsyncClient())

    papers = client._parse_feed(SAMPLE_FEED)

    assert len(papers) == 1
    paper = papers[0]
    assert paper.paper_key == "arxiv:2101.00001"
    assert paper.title == "A Great Paper About Retrieval Augmented Generation"
    assert paper.authors == ["Ada Lovelace", "Alan Turing"]
    assert paper.published == "2021-01-01"
    assert paper.pdf_url == "http://arxiv.org/pdf/2101.00001v2"
    assert paper.source.value == "arxiv"


def test_parse_feed_handles_malformed_xml_gracefully():
    settings = Settings(anthropic_api_key="test")
    client = ArxivClient(settings, httpx.AsyncClient())

    papers = client._parse_feed("not xml at all")

    assert papers == []


def test_parse_feed_skips_entries_missing_id():
    settings = Settings(anthropic_api_key="test")
    client = ArxivClient(settings, httpx.AsyncClient())
    feed = """<feed xmlns="http://www.w3.org/2005/Atom">
      <entry><title>No id here</title></entry>
    </feed>"""

    papers = client._parse_feed(feed)

    assert papers == []
