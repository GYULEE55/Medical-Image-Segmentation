"""
Medical RAG 체인 모듈

LangChain LCEL 기반 RAG 파이프라인:
  질문 → ChromaDB 검색 → GPT-4o-mini 답변 생성 → 출처 반환

사용법:
    from rag.chain import MedicalRAGChain
    rag = MedicalRAGChain()
    result = await rag.query("폴립이란 무엇인가?")
"""

import os
from pathlib import Path
from typing import Optional

# chain.py가 독립적으로 임포트되어도 API 키를 읽을 수 있도록
# app.py의 load_dotenv와 별개로 여기서도 로드 (중복 호출해도 문제없음)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# ── 경로 설정 ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "medical_knowledge"

# ── 의료 특화 시스템 프롬프트 ──────────────────────────────────
# 핵심 설계 원칙:
# 1. 제공된 문서 컨텍스트만 사용 → hallucination 방지
# 2. 출처 명시 요구 → 의료 정보 신뢰성 확보
# 3. 모르면 모른다고 답변 → 잘못된 의료 정보 차단
MEDICAL_SYSTEM_PROMPT = """당신은 의료 지식 전문 AI 어시스턴트입니다.
반드시 아래 제공된 의료 문서 컨텍스트만을 기반으로 질문에 답변하세요.

규칙:
1. 컨텍스트에 없는 정보는 "제공된 문서에서 해당 정보를 찾을 수 없습니다"라고 답하세요.
2. 의료 정보는 반드시 어떤 문서에서 나온 정보인지 언급하세요.
3. 진단이나 치료를 직접 권고하지 마세요. 반드시 의료 전문가 상담을 권장하세요.
4. 한국어로 명확하고 이해하기 쉽게 답변하세요.

<context>
{context}
</context>"""


class MedicalRAGChain:
    """의료 지식 RAG 체인 (싱글톤 패턴으로 서버에서 1회 초기화)"""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        top_k: int = 5,
        device: str = "cpu",
    ):
        """
        Args:
            openai_api_key: OpenAI API 키 (없으면 환경변수 OPENAI_API_KEY 사용)
            top_k: 검색할 문서 청크 수
            device: 임베딩 모델 실행 디바이스 ("cpu" or "cuda")
        """
        self.top_k = top_k
        self._chain = None
        self._vector_store = None

        # API 키 설정
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        self._initialize()

    def _initialize(self):
        """RAG 체인 초기화 (임베딩 모델 + 벡터스토어 + LLM + 체인)"""
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDB가 없습니다: {CHROMA_DIR}\n"
                "먼저 'python rag/ingest.py'를 실행해 문서를 인덱싱하세요."
            )

        print("[RAG] 임베딩 모델 로드 중 (BAAI/bge-m3)...")
        # BGE-M3: 한국어+영어 의료 문서 동시 처리 가능
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )

        print("[RAG] ChromaDB 로드 중...")
        self._vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

        # 검색기: 코사인 유사도 기반 상위 k개 청크 반환
        retriever = self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k},
        )

        # LLM: GPT-4o-mini (저렴 + 충분한 품질)
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,  # 의료 정보는 일관성이 중요 → temperature=0
        )

        # LCEL 체인 조립 (현대적 패턴)
        # RetrievalQA.from_chain_type() 는 deprecated → create_retrieval_chain 사용
        prompt = ChatPromptTemplate.from_messages([
            ("system", MEDICAL_SYSTEM_PROMPT),
            ("human", "{input}"),
        ])
        qa_chain = create_stuff_documents_chain(llm, prompt)
        self._chain = create_retrieval_chain(retriever, qa_chain)

        print("[RAG] 초기화 완료 ✅")

    async def query(self, question: str) -> dict:
        """
        의료 지식 RAG 쿼리 (비동기)

        Args:
            question: 질문 텍스트

        Returns:
            {
                "answer": str,           # LLM 답변
                "sources": list[dict],   # 출처 문서 목록
                "num_sources": int,      # 검색된 청크 수
            }
        """
        if self._chain is None:
            raise RuntimeError("RAG 체인이 초기화되지 않았습니다.")

        # ainvoke: async 버전 (FastAPI async endpoint에서 필수)
        response = await self._chain.ainvoke({"input": question})

        # 출처 정보 추출
        sources = []
        for doc in response.get("context", []):
            sources.append({
                "source_file": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "content_preview": doc.page_content[:200] + "...",
            })

        return {
            "answer": response["answer"],
            "sources": sources,
            "num_sources": len(sources),
        }

    def query_sync(self, question: str) -> dict:
        """동기 버전 쿼리 (테스트/스크립트용)"""
        if self._chain is None:
            raise RuntimeError("RAG 체인이 초기화되지 않았습니다.")

        response = self._chain.invoke({"input": question})

        sources = []
        for doc in response.get("context", []):
            sources.append({
                "source_file": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", "N/A"),
                "content_preview": doc.page_content[:200] + "...",
            })

        return {
            "answer": response["answer"],
            "sources": sources,
            "num_sources": len(sources),
        }

    @property
    def is_ready(self) -> bool:
        """RAG 체인 준비 상태 확인"""
        return self._chain is not None


# ── 로컬 테스트 ────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    print("Medical RAG 체인 테스트")
    print("-" * 40)

    rag = MedicalRAGChain()

    test_questions = [
        "위장 폴립이란 무엇인가요?",
        "대장내시경 검사 전 준비사항은?",
        "폴립 제거 후 주의사항은?",
    ]

    for q in test_questions:
        print(f"\n질문: {q}")
        result = rag.query_sync(q)
        print(f"답변: {result['answer'][:300]}...")
        print(f"출처: {result['num_sources']}개 문서")
        for src in result["sources"][:2]:
            print(f"  - {src['source_file']} p.{src['page']}")
