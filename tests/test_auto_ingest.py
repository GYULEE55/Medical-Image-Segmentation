from pathlib import Path

from rag.auto_ingest import parse_pubmed_xml, save_articles

SAMPLE_XML = """
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>Colonoscopy outcomes in high-risk patients</ArticleTitle>
        <Journal>
          <JournalIssue>
            <PubDate>
              <Year>2024</Year>
              <Month>Jun</Month>
              <Day>15</Day>
            </PubDate>
          </JournalIssue>
          <Title>Clinical Endoscopy</Title>
        </Journal>
        <AuthorList>
          <Author>
            <LastName>Kim</LastName>
            <Initials>JH</Initials>
          </Author>
        </AuthorList>
        <Abstract>
          <AbstractText Label="Background">Background text.</AbstractText>
          <AbstractText>Methods text.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
""".strip()


def test_parse_pubmed_xml_extracts_required_fields():
    rows = parse_pubmed_xml(SAMPLE_XML)
    assert len(rows) == 1

    row = rows[0]
    assert row["pmid"] == "12345678"
    assert row["title"] == "Colonoscopy outcomes in high-risk patients"
    assert row["journal"] == "Clinical Endoscopy"
    assert row["pub_date"] == "2024-Jun-15"
    assert "[Background] Background text." in row["abstract"]
    assert "Methods text." in row["abstract"]
    assert row["url"] == "https://pubmed.ncbi.nlm.nih.gov/12345678/"


def test_save_articles_writes_txt_and_skips_duplicates(tmp_path: Path):
    rows = [
        {
            "pmid": "12345678",
            "title": "Sample title",
            "journal": "Sample journal",
            "pub_date": "2024",
            "authors": ["Kim JH"],
            "abstract": "Sample abstract",
            "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
        }
    ]

    created1, skipped1 = save_articles(rows, tmp_path)
    created2, skipped2 = save_articles(rows, tmp_path)

    assert created1 == 1
    assert skipped1 == 0
    assert created2 == 0
    assert skipped2 == 1
    assert (tmp_path / "pubmed_12345678.txt").exists()
