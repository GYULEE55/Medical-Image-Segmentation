"""
의료 문서 인덱싱 파이프라인 (1회 실행)

PDF 문서 → 텍스트 추출 → 청킹 → BGE-M3 임베딩 → ChromaDB 저장

사용법:
    python rag/ingest.py

rag/docs/ 폴더에 PDF 파일을 넣고 실행하면
rag/vectorstore/ 에 ChromaDB가 생성됩니다.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "docs"           # PDF 파일 저장 폴더
CHROMA_DIR = BASE_DIR / "vectorstore" # ChromaDB 저장 폴더
COLLECTION_NAME = "medical_knowledge"

# ── 청킹 설정 ──────────────────────────────────────────────────
# 의료 문서 최적화:
# - chunk_size=512: 의료 용어/문장 맥락 보존 (너무 작으면 맥락 손실)
# - chunk_overlap=64: 청크 경계에서 정보 손실 방지
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def load_pdfs(pdf_dir: Path) -> list:
    """PDF 폴더에서 모든 문서 로드 (페이지별 메타데이터 포함)"""
    all_docs = []
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"[경고] {pdf_dir} 에 PDF 파일이 없습니다.")
        print("  → rag/docs/ 폴더에 의료 문서 PDF를 넣어주세요.")
        return []

    for pdf_path in pdf_files:
        print(f"  로딩: {pdf_path.name}")
        loader = PyMuPDFLoader(
            str(pdf_path),
            mode="page",  # 페이지별 Document 생성 → 출처 추적 용이
        )
        pages = loader.load()
        all_docs.extend(pages)
        print(f"    → {len(pages)} 페이지")

    return all_docs


def chunk_documents(documents: list) -> list:
    """문서를 청크로 분할 (의료 문서 최적화 설정)"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # 단락 → 문장 → 단어 순서로 분할 (의미 단위 보존)
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    return chunks


def build_vectorstore(chunks: list) -> Chroma:
    """BGE-M3 임베딩으로 ChromaDB 벡터스토어 구축"""
    print("\n[임베딩 모델 로드] BAAI/bge-m3 ...")
    print("  (첫 실행 시 모델 다운로드 ~1.5GB, 시간 소요)")

    # BGE-M3: 다국어 지원 (한국어+영어 의료 문서 동시 처리 가능)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={
            "device": "cpu",           # GPU 있으면 "cuda"
            "trust_remote_code": True,
        },
        encode_kwargs={
            "normalize_embeddings": True,  # 코사인 유사도 계산에 필수
        },
    )

    print(f"\n[ChromaDB 저장] {CHROMA_DIR} ...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    return vector_store


def main():
    print("=" * 50)
    print("의료 문서 인덱싱 파이프라인")
    print("=" * 50)

    # 1. PDF 로딩
    print(f"\n[1/3] PDF 로딩: {PDF_DIR}")
    documents = load_pdfs(PDF_DIR)
    if not documents:
        return
    print(f"  → 총 {len(documents)} 페이지 로드 완료")

    # 2. 청킹
    print(f"\n[2/3] 청킹 (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    chunks = chunk_documents(documents)
    print(f"  → 총 {len(chunks)} 청크 생성")

    # 3. 벡터스토어 구축
    print("\n[3/3] 벡터스토어 구축")
    vector_store = build_vectorstore(chunks)

    print("\n" + "=" * 50)
    print("✅ 인덱싱 완료!")
    print(f"   저장 위치: {CHROMA_DIR}")
    print(f"   총 청크 수: {len(chunks)}")
    print("\n다음 단계: python -m uvicorn api.app:app --reload")
    print("=" * 50)


if __name__ == "__main__":
    main()
