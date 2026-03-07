import os
from pathlib import Path
from typing import Dict

MODEL_PATHS: Dict[str, str] = {
    "polyp": os.getenv("MODEL_PATH", str(Path(__file__).parent.parent / "best.pt")),
    "dental": os.getenv(
        "DENTAL_MODEL_PATH", str(Path(__file__).parent.parent / "best_dentex.pt")
    ),
}

VLM_MAX_EDGE = int(os.getenv("VLM_MAX_EDGE", "960"))
VLM_JPEG_QUALITY = int(os.getenv("VLM_JPEG_QUALITY", "75"))
VLM_TIMEOUT_SECONDS = float(os.getenv("VLM_TIMEOUT_SECONDS", "180"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "10485760"))
WEB_EVIDENCE_TIMEOUT_SECONDS = float(os.getenv("WEB_EVIDENCE_TIMEOUT_SECONDS", "8"))
WEB_EVIDENCE_MAX_ARTICLES = int(os.getenv("WEB_EVIDENCE_MAX_ARTICLES", "3"))
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

NO_EVIDENCE_TEXT = "제공된 문서에서 해당 정보를 찾을 수 없습니다."

class_names_kr = {
    "polyp": "위장 폴립",
    "Impacted": "매복치",
    "Caries": "충치",
    "Deep Caries": "깊은 충치",
    "Periapical Lesion": "치근단 병변",
}

SAMPLE_REGISTRY = {
    "colon_polyp": {
        "file": "colon_polyp.jpg",
        "model_type": "polyp",
        "label": "대장 내시경 (폴립)",
    },
    "dental_xray": {
        "file": "dental_xray.png",
        "model_type": "dental",
        "label": "치과 파노라마 X-ray",
    },
}
