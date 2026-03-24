import argparse
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "docs" / "auto"
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
PUBMED_TOOL = "MedicalAISegRAG"
DEFAULT_TOPICS = [
    "colonoscopy polyp guideline",
    "endoscopic mucosal resection polyp",
    "gastrointestinal polyp surveillance",
]


def _safe_filename(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", text)


def _collect_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _extract_pub_date(article: ET.Element) -> str:
    pub_date = article.find(".//Journal/JournalIssue/PubDate")
    if pub_date is None:
        return "N/A"

    year = _collect_text(pub_date.find("Year"))
    month = _collect_text(pub_date.find("Month"))
    day = _collect_text(pub_date.find("Day"))
    medline_date = _collect_text(pub_date.find("MedlineDate"))

    if year:
        pieces = [year]
        if month:
            pieces.append(month)
        if day:
            pieces.append(day)
        return "-".join(pieces)
    if medline_date:
        return medline_date
    return "N/A"


def _extract_abstract(article: ET.Element) -> str:
    abstract_nodes = article.findall(".//Abstract/AbstractText")
    if not abstract_nodes:
        return ""

    parts: list[str] = []
    for node in abstract_nodes:
        label = (node.attrib.get("Label") or "").strip()
        section_text = _collect_text(node)
        if not section_text:
            continue
        if label:
            parts.append(f"[{label}] {section_text}")
        else:
            parts.append(section_text)

    return "\n".join(parts).strip()


def parse_pubmed_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    rows: list[dict] = []

    for entry in root.findall(".//PubmedArticle"):
        pmid = _collect_text(entry.find(".//PMID"))
        article = entry.find(".//Article")
        if not pmid or article is None:
            continue

        title = _collect_text(article.find("ArticleTitle"))
        journal = _collect_text(article.find(".//Journal/Title"))
        abstract = _extract_abstract(article)
        pub_date = _extract_pub_date(article)

        authors = []
        for author in article.findall(".//AuthorList/Author"):
            last_name = _collect_text(author.find("LastName"))
            initials = _collect_text(author.find("Initials"))
            collective = _collect_text(author.find("CollectiveName"))
            if collective:
                authors.append(collective)
            elif last_name:
                if initials:
                    authors.append(f"{last_name} {initials}")
                else:
                    authors.append(last_name)

        rows.append(
            {
                "pmid": pmid,
                "title": title or "Untitled",
                "journal": journal or "Unknown Journal",
                "pub_date": pub_date,
                "authors": authors,
                "abstract": abstract,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )

    return rows


def _request_with_retry(
    client: httpx.Client,
    url: str,
    params: dict,
    retries: int = 3,
) -> httpx.Response:
    delay = 0.8
    for attempt in range(retries):
        response = client.get(url, params=params)
        if response.status_code in (429, 500, 502, 503, 504):
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
        response.raise_for_status()
        return response
    raise RuntimeError("PubMed 요청 실패")


def _base_params(email: str, api_key: str) -> dict:
    params = {
        "tool": PUBMED_TOOL,
        "email": email,
    }
    if api_key:
        params["api_key"] = api_key
    return params


def search_pubmed_ids(
    client: httpx.Client,
    query: str,
    max_results: int,
    email: str,
    api_key: str,
    days: int | None,
) -> list[str]:
    params = {
        **_base_params(email=email, api_key=api_key),
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(max_results),
        "sort": "relevance",
    }
    if days is not None:
        params["reldate"] = str(days)
        params["datetype"] = "pdat"

    response = _request_with_retry(client=client, url=PUBMED_ESEARCH_URL, params=params)
    payload = response.json()
    id_list = payload.get("esearchresult", {}).get("idlist", [])
    return [str(x) for x in id_list if str(x).strip()]


def fetch_pubmed_details(
    client: httpx.Client, ids: list[str], email: str, api_key: str
) -> list[dict]:
    if not ids:
        return []

    params = {
        **_base_params(email=email, api_key=api_key),
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml",
    }
    response = _request_with_retry(client=client, url=PUBMED_EFETCH_URL, params=params)
    return parse_pubmed_xml(response.text)


def save_articles(articles: list[dict], output_dir: Path) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    for row in articles:
        abstract = (row.get("abstract") or "").strip()
        if not abstract:
            skipped += 1
            continue

        pmid = row["pmid"]
        path = output_dir / f"pubmed_{_safe_filename(pmid)}.txt"
        if path.exists():
            skipped += 1
            continue

        content = [
            f"Title: {row['title']}",
            f"PMID: {pmid}",
            f"Journal: {row['journal']}",
            f"Published: {row['pub_date']}",
            f"URL: {row['url']}",
            f"Authors: {', '.join(row['authors']) if row['authors'] else 'N/A'}",
            "",
            "Abstract:",
            abstract,
        ]
        path.write_text("\n".join(content), encoding="utf-8")
        created += 1

    return created, skipped


def run_ingest(project_root: Path) -> None:
    ingest_script = project_root / "rag" / "ingest.py"
    subprocess.run(
        [sys.executable, str(ingest_script)], check=True, cwd=str(project_root)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topics",
        nargs="+",
        default=DEFAULT_TOPICS,
        help="PubMed 검색 토픽 목록 (공백 구분)",
    )
    parser.add_argument(
        "--max-results", type=int, default=15, help="토픽별 최대 PMID 수"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=3650,
        help="최근 n일 이내 논문만 검색 (예: 365=1년, 3650=10년)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="문서 저장 폴더",
    )
    parser.add_argument(
        "--email",
        type=str,
        default="",
        help="NCBI 권장 식별용 이메일 (없으면 PUBMED_EMAIL 환경변수 사용)",
    )
    parser.add_argument(
        "--no-reindex",
        action="store_true",
        help="수집만 하고 rag/ingest.py는 실행하지 않음",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(BASE_DIR.parent / ".env", override=True)
    args = parse_args()
    email = (args.email or "").strip()
    if not email:
        email = os.environ.get("PUBMED_EMAIL", "").strip()
    if not email:
        raise ValueError("PUBMED_EMAIL 환경변수 또는 --email 인자를 지정하세요.")
    api_key = os.environ.get("PUBMED_API_KEY", "").strip()

    output_dir = Path(args.output_dir)
    project_root = BASE_DIR.parent
    total_created = 0
    total_skipped = 0
    seen_pmids: set[str] = set()

    print("=" * 60)
    print("PubMed 자동 수집 시작")
    print("=" * 60)
    print(f"토픽 수: {len(args.topics)}")
    print(f"토픽별 최대 PMID: {args.max_results}")
    print(f"저장 경로: {output_dir}")

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for topic in args.topics:
            print(f"\n[검색] {topic}")
            ids = search_pubmed_ids(
                client=client,
                query=topic,
                max_results=args.max_results,
                email=email,
                api_key=api_key,
                days=args.days,
            )
            ids = [x for x in ids if x not in seen_pmids]
            seen_pmids.update(ids)
            print(f"  PMID 후보: {len(ids)}개")

            if not ids:
                continue

            details = fetch_pubmed_details(
                client=client,
                ids=ids,
                email=email,
                api_key=api_key,
            )
            created, skipped = save_articles(details, output_dir)
            total_created += created
            total_skipped += skipped
            print(f"  저장 완료: +{created}, 건너뜀: {skipped}")

            time.sleep(0.34)

    print("\n" + "=" * 60)
    print("수집 요약")
    print(f"신규 저장: {total_created}")
    print(f"건너뜀(중복/초록없음): {total_skipped}")
    print("=" * 60)

    if args.no_reindex:
        print("\n--no-reindex 옵션으로 인덱싱은 건너뜀")
        return

    print("\n[인덱싱] rag/ingest.py 실행")
    run_ingest(project_root)
    print("✅ 자동 수집 + 인덱싱 완료")


if __name__ == "__main__":
    main()
