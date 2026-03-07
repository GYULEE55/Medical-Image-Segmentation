import os
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Optional

import cv2
import httpx
import numpy as np
from fastapi import HTTPException

from . import state
from .constants import (
    MAX_UPLOAD_BYTES,
    PUBMED_EFETCH_URL,
    PUBMED_ESEARCH_URL,
    VLM_JPEG_QUALITY,
    VLM_MAX_EDGE,
    WEB_EVIDENCE_MAX_ARTICLES,
    WEB_EVIDENCE_TIMEOUT_SECONDS,
)


def observe_inference(endpoint: str, model_type: str, started_at: float) -> float:
    elapsed = time.perf_counter() - started_at
    state.MODEL_INFERENCE_TOTAL.labels(endpoint=endpoint, model_type=model_type).inc()
    state.MODEL_INFERENCE_DURATION_SECONDS.labels(endpoint=endpoint, model_type=model_type).observe(
        elapsed
    )
    try:
        import psutil

        process = psutil.Process(os.getpid())
        state.PROCESS_MEMORY_BYTES.set(process.memory_info().rss)
    except Exception:
        pass
    return elapsed


def prepare_vlm_image_bytes(img: np.ndarray, original_bytes: bytes):
    h, w = img.shape[:2]
    longest = max(w, h)

    if longest <= VLM_MAX_EDGE:
        return original_bytes, {"resized": False, "width": w, "height": h}

    scale = VLM_MAX_EDGE / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), VLM_JPEG_QUALITY])

    if not ok:
        return original_bytes, {"resized": False, "width": w, "height": h}

    return encoded.tobytes(), {"resized": True, "width": new_w, "height": new_h}


def validate_upload_size(contents: bytes) -> None:
    if not contents:
        raise HTTPException(400, "빈 파일은 처리할 수 없습니다")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            f"파일이 너무 큽니다. 최대 {MAX_UPLOAD_BYTES} bytes 까지 허용됩니다",
        )


def compact_text(text: Any, max_len: int = 220) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if not compact:
        return ""
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def extract_pubmed_abstract(article: ET.Element) -> str:
    abstract_nodes = article.findall(".//Abstract/AbstractText")
    if not abstract_nodes:
        return ""

    sections = []
    for node in abstract_nodes:
        section_text = "".join(node.itertext()).strip()
        if section_text:
            sections.append(section_text)
    return "\n".join(sections).strip()


def extract_pubmed_year(article: ET.Element) -> str:
    pub_date = article.find(".//Journal/JournalIssue/PubDate")
    if pub_date is None:
        return "N/A"
    year = (pub_date.findtext("Year") or "").strip()
    if year:
        return year
    medline_date = (pub_date.findtext("MedlineDate") or "").strip()
    return medline_date[:4] if medline_date else "N/A"


def parse_pubmed_articles(xml_text: str) -> list[dict[str, Any]]:
    root = ET.fromstring(xml_text)
    rows: list[dict[str, Any]] = []

    for entry in root.findall(".//PubmedArticle"):
        pmid = (entry.findtext(".//PMID") or "").strip()
        article = entry.find(".//Article")
        if not pmid or article is None:
            continue

        title = "".join((article.find("ArticleTitle") or ET.Element("x")).itertext()).strip()
        journal = (article.findtext(".//Journal/Title") or "").strip() or "Unknown Journal"
        abstract = extract_pubmed_abstract(article)
        year = extract_pubmed_year(article)

        rows.append(
            {
                "pmid": pmid,
                "title": title or "Untitled",
                "journal": journal,
                "year": year,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "abstract": abstract,
            }
        )
    return rows


def topic_seed_by_model(model_type: str) -> str:
    if model_type == "dental":
        return "dental caries periapical lesion impacted tooth panoramic x-ray guideline"
    return "colorectal polyp colonoscopy surveillance guideline endoscopic management"


async def fetch_pubmed_web_evidence(query: str, model_type: str) -> Optional[dict[str, Any]]:
    search_query = f"{topic_seed_by_model(model_type)} {query[:220]}"
    timeout = httpx.Timeout(WEB_EVIDENCE_TIMEOUT_SECONDS)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            esearch_params = {
                "db": "pubmed",
                "term": search_query,
                "retmode": "json",
                "retmax": str(WEB_EVIDENCE_MAX_ARTICLES),
                "sort": "relevance",
            }
            esearch_res = await client.get(PUBMED_ESEARCH_URL, params=esearch_params)
            esearch_res.raise_for_status()
            ids = esearch_res.json().get("esearchresult", {}).get("idlist", [])
            pmids = [str(x).strip() for x in ids if str(x).strip()]
            if not pmids:
                return None

            efetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            }
            efetch_res = await client.get(PUBMED_EFETCH_URL, params=efetch_params)
            efetch_res.raise_for_status()
            articles = parse_pubmed_articles(efetch_res.text)
    except Exception:
        return None

    if not articles:
        return None

    evidence_lines = []
    sources = []
    for idx, article in enumerate(articles, start=1):
        preview = compact_text(article.get("abstract") or article.get("title"), max_len=260)
        evidence_lines.append(
            f"{idx}) {article['title']} ({article['journal']}, {article['year']})\n- 핵심: {preview}\n- URL: {article['url']}"
        )
        sources.append(
            {
                "source_file": f"PubMed:{article['pmid']}",
                "page": "web",
                "content_preview": preview,
                "url": article["url"],
            }
        )

    answer = "\n\n".join(evidence_lines)
    return {
        "query_used": compact_text(search_query, max_len=200),
        "query_source": "web_fallback",
        "answer": answer,
        "sources": sources,
        "num_sources": len(sources),
        "grounded": True,
        "reason": "web_pubmed_fallback",
        "disclaimer": "인터넷(PubMed) 검색 기반 참고 정보입니다. 최종 의료 판단은 전문의 상담이 필요합니다.",
    }
