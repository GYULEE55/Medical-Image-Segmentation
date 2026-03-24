"""
Medical RAG 체인 모듈 — LangGraph 버전

LangGraph StateGraph 기반 RAG 파이프라인:
  질문 → retrieve → (조건부) generate → format_sources → 최종 응답

노드 구성:
  retrieve        : ChromaDB 벡터 검색 + relevance score 필터링
  generate        : LLM 답변 생성 (MEDICAL_SYSTEM_PROMPT 사용)
  format_sources  : 출처 목록 정제
  no_evidence     : 근거 없음 고정 응답 반환

조건부 엣지:
  retrieve → check_relevance → no_evidence (문서 없음) or generate
  generate → check_answer   → no_evidence (환각 감지)  or format_sources

외부 인터페이스는 LCEL 버전과 동일:
  result = await rag.query("폴립이란?")
  # → {"answer": str, "sources": [...], "num_sources": int}

면접 포인트:
  - LangGraph: 복잡한 RAG 제어 흐름(분기/반복)을 그래프로 명시적으로 표현
  - StateGraph: 각 노드가 State를 받아 업데이트 → 불변 상태 추적 가능
  - 조건부 엣지: no-evidence 가드를 코드가 아닌 그래프 구조로 표현
  - LCEL vs LangGraph: LCEL은 선형 파이프라인, LangGraph는 비선형 워크플로우
"""

import os
from pathlib import Path
from typing import Optional, cast

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

from langgraph.graph import END, StateGraph

from core.structured_logging import get_logger

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

logger = get_logger("rag.chain")

# ── 경로 / 환경변수 설정 ───────────────────────────────────────
BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "medical_knowledge"

RAG_LLM_PROVIDER = os.getenv("RAG_LLM_PROVIDER", "ollama").lower()
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
RAG_OPENAI_MODEL = os.getenv("RAG_OPENAI_MODEL", "gpt-4o-mini")
RAG_OLLAMA_MODEL = os.getenv("RAG_OLLAMA_MODEL", "llama3.1:8b")
RAG_OLLAMA_BASE_URL = os.getenv(
    "RAG_OLLAMA_BASE_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434")
)
RAG_OLLAMA_NUM_PREDICT = int(os.getenv("RAG_OLLAMA_NUM_PREDICT", "512"))
RAG_MIN_RELEVANCE = float(os.getenv("RAG_MIN_RELEVANCE", "0.2"))
RAG_USE_OPENAI_FALLBACK = os.getenv("RAG_USE_OPENAI_FALLBACK", "true").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

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

NO_EVIDENCE_ANSWER = "제공된 문서에서 해당 정보를 찾을 수 없습니다."


# ── LangGraph State 정의 ───────────────────────────────────────
# TypedDict로 그래프 전체에서 공유되는 상태를 정의
# 각 노드는 State를 받아서 변경할 필드만 dict로 반환
class SourceEntry(TypedDict):
    source_file: str
    page: str
    content_preview: str


class QueryResult(TypedDict):
    answer: str
    sources: list[SourceEntry]
    num_sources: int


StateUpdate = dict[str, object]


class RAGState(TypedDict):
    """RAG 그래프 상태"""

    question: str  # 입력 질문
    filtered_docs: list[Document]  # relevance 필터링 통과한 문서
    answer: str  # LLM 생성 답변
    sources: list[SourceEntry]  # 포맷팅된 출처 목록
    num_sources: int  # 출처 수
    is_no_evidence: bool  # no-evidence 가드 발동 여부


class MedicalRAGChain:
    """
    의료 지식 RAG 체인 — LangGraph StateGraph 기반

    그래프 구조:
        START
          ↓
        retrieve ──── (문서 없음) ────→ no_evidence → END
          ↓ (문서 있음)
        generate ──── (환각 감지) ────→ no_evidence → END
          ↓ (정상 답변)
        format_sources → END

    외부 인터페이스는 LCEL 버전과 동일하게 유지:
        result = await rag.query("질문")
        result = rag.query_sync("질문")
        rag.is_ready
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        top_k: Optional[int] = None,
        device: str = "cpu",
        provider: Optional[str] = None,
    ):
        self.top_k = top_k if top_k is not None else RAG_TOP_K
        self.min_relevance = RAG_MIN_RELEVANCE
        self.provider = (provider or RAG_LLM_PROVIDER).lower()
        self._graph = None
        self._vector_store = None

        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        self._initialize()

    # ── LLM 빌드 ──────────────────────────────────────────────
    def _build_llm(self):
        """LLM 선택: openai 직접 or ollama (OpenAI fallback 포함)"""
        if self.provider == "openai":
            return ChatOpenAI(model=RAG_OPENAI_MODEL, temperature=0)

        if self.provider != "ollama":
            raise ValueError(
                f"지원하지 않는 RAG_LLM_PROVIDER: {self.provider} (openai/ollama 사용 가능)"
            )

        if ChatOllama is None:
            raise ImportError(
                "langchain_ollama 패키지가 없습니다. requirements 설치 후 재시작하세요."
            )

        local_llm = ChatOllama(
            model=RAG_OLLAMA_MODEL,
            base_url=RAG_OLLAMA_BASE_URL,
            temperature=0,
            num_predict=RAG_OLLAMA_NUM_PREDICT,
        )

        # Ollama 실패 시 OpenAI로 자동 전환 (resilience 패턴)
        if RAG_USE_OPENAI_FALLBACK and os.getenv("OPENAI_API_KEY"):
            cloud_llm = ChatOpenAI(model=RAG_OPENAI_MODEL, temperature=0)
            return local_llm.with_fallbacks([cloud_llm])

        return local_llm

    # ── 초기화: 벡터스토어 + LLM + 그래프 빌드 ───────────────
    def _initialize(self):
        """임베딩 모델 로드 → 벡터스토어 연결 → LangGraph 그래프 컴파일"""
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                f"ChromaDB가 없습니다: {CHROMA_DIR}\n"
                "먼저 'python rag/ingest.py'를 실행해 문서를 인덱싱하세요."
            )

        logger.info("embedding_model_loading", stage="rag", model="BAAI/bge-m3")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu", "trust_remote_code": True},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("vectorstore_loading", stage="rag", path=str(CHROMA_DIR))
        self._vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

        llm = self._build_llm()
        logger.info("llm_provider_selected", stage="rag", provider=self.provider)

        # LangGraph 그래프 빌드
        self._graph = self._build_graph(llm)
        logger.info("langgraph_rag_initialized", stage="rag")

    # ── 그래프 노드 정의 ──────────────────────────────────────

    def _make_retrieve_node(self):
        """
        retrieve 노드: ChromaDB 벡터 검색 + relevance score 필터링

        LCEL 버전의 _retrieve_docs() 에 해당.
        State에서 question을 받아 filtered_docs를 업데이트.
        """
        vector_store = self._vector_store
        if vector_store is None:
            raise RuntimeError("RAG vector store is not initialized")
        vector_store = cast(Chroma, vector_store)
        top_k = self.top_k
        min_relevance = self.min_relevance

        def retrieve(state: RAGState) -> StateUpdate:
            question = state["question"]
            docs_with_scores = vector_store.similarity_search_with_relevance_scores(
                question, k=top_k
            )
            # relevance score 0.2 미만 제거 → no-evidence 가드 1단계
            filtered = [
                doc
                for doc, score in docs_with_scores
                if score is not None and score >= min_relevance
            ]
            logger.info(
                "retrieve_completed",
                stage="rag",
                total=len(docs_with_scores),
                filtered=len(filtered),
                min_relevance=min_relevance,
            )
            return {"filtered_docs": filtered}

        return retrieve

    def _make_generate_node(self, llm):
        """
        generate 노드: 필터링된 문서를 컨텍스트로 LLM 답변 생성

        LCEL 버전의 _qa_chain.ainvoke() 에 해당.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", MEDICAL_SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )
        chain = prompt | llm

        async def generate(state: RAGState) -> StateUpdate:
            question = state["question"]
            docs = state["filtered_docs"]

            # 문서 내용을 컨텍스트 문자열로 결합
            context_text = "\n\n---\n\n".join(doc.page_content for doc in docs)

            response = await chain.ainvoke(
                {
                    "input": question,
                    "context": context_text,
                }
            )
            # ChatOpenAI / ChatOllama 둘 다 .content 속성 사용
            answer = response.content if hasattr(response, "content") else str(response)

            logger.info("generate_completed", stage="rag", answer_len=len(answer))
            return {"answer": answer}

        return generate

    def _make_generate_node_sync(self, llm):
        """generate 노드 동기 버전 (query_sync용)"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", MEDICAL_SYSTEM_PROMPT),
                ("human", "{input}"),
            ]
        )
        chain = prompt | llm

        def generate_sync(state: RAGState) -> StateUpdate:
            question = state["question"]
            docs = state["filtered_docs"]
            context_text = "\n\n---\n\n".join(doc.page_content for doc in docs)

            response = chain.invoke({"input": question, "context": context_text})
            answer = response.content if hasattr(response, "content") else str(response)
            return {"answer": answer}

        return generate_sync

    @staticmethod
    def _format_sources_node(state: RAGState) -> StateUpdate:
        """
        format_sources 노드: 문서 메타데이터를 출처 딕셔너리로 정제

        LCEL 버전의 _format_sources() 에 해당.
        """
        sources: list[SourceEntry] = []
        for doc in state["filtered_docs"]:
            raw_page = doc.metadata.get("page", "N/A")
            source_path = doc.metadata.get("source", "unknown")
            source_file = Path(source_path).name if source_path else "unknown"
            sources.append(
                {
                    "source_file": source_file,
                    "page": str(raw_page) if raw_page is not None else "N/A",
                    "content_preview": doc.page_content[:200] + "...",
                }
            )
        return {"sources": sources, "num_sources": len(sources)}

    @staticmethod
    def _no_evidence_node(state: RAGState) -> StateUpdate:
        """
        no_evidence 노드: 근거 없음 고정 응답

        LCEL 버전의 _no_evidence_payload() 에 해당.
        조건부 엣지로 라우팅되어 LLM 호출 없이 즉시 반환.
        """
        logger.info("no_evidence_triggered", stage="rag")
        return {
            "answer": NO_EVIDENCE_ANSWER,
            "sources": [],
            "num_sources": 0,
            "is_no_evidence": True,
        }

    # ── 조건부 엣지 (라우팅 함수) ─────────────────────────────

    @staticmethod
    def _check_relevance(state: RAGState) -> str:
        """
        retrieve 이후 조건부 엣지:
        - 문서 없음 → "no_evidence" 노드로
        - 문서 있음 → "generate" 노드로
        """
        if not state["filtered_docs"]:
            return "no_evidence"
        return "generate"

    @staticmethod
    def _check_answer(state: RAGState) -> str:
        """
        generate 이후 조건부 엣지:
        - LLM이 "제공된 문서에서..." 반환 → "no_evidence" 노드로
        - 정상 답변 → "format_sources" 노드로
        """
        if NO_EVIDENCE_ANSWER in state.get("answer", ""):
            return "no_evidence"
        return "format_sources"

    # ── 그래프 빌드 ───────────────────────────────────────────

    def _build_graph(self, llm):
        """
        LangGraph StateGraph 조립

        그래프 구조:
            retrieve
              ↓ (조건부)
            check_relevance ── 없음 ──→ no_evidence → END
              ↓ 있음
            generate
              ↓ (조건부)
            check_answer ──── 환각 ──→ no_evidence → END
              ↓ 정상
            format_sources → END
        """
        builder = StateGraph(RAGState)

        # 노드 등록
        builder.add_node("retrieve", self._make_retrieve_node())
        builder.add_node("generate", self._make_generate_node(llm))
        builder.add_node("format_sources", self._format_sources_node)
        builder.add_node("no_evidence", self._no_evidence_node)

        # 진입점
        builder.set_entry_point("retrieve")

        # 조건부 엣지 1: retrieve → check_relevance → no_evidence or generate
        builder.add_conditional_edges(
            "retrieve",
            self._check_relevance,
            {"no_evidence": "no_evidence", "generate": "generate"},
        )

        # 조건부 엣지 2: generate → check_answer → no_evidence or format_sources
        builder.add_conditional_edges(
            "generate",
            self._check_answer,
            {"no_evidence": "no_evidence", "format_sources": "format_sources"},
        )

        # 종료 엣지
        builder.add_edge("format_sources", END)
        builder.add_edge("no_evidence", END)

        return builder.compile()

    # ── 동기 그래프 (query_sync 전용) ─────────────────────────

    def _build_sync_graph(self, llm):
        """query_sync용 동기 그래프 (generate 노드만 동기 버전)"""
        builder = StateGraph(RAGState)

        builder.add_node("retrieve", self._make_retrieve_node())
        builder.add_node("generate", self._make_generate_node_sync(llm))
        builder.add_node("format_sources", self._format_sources_node)
        builder.add_node("no_evidence", self._no_evidence_node)

        builder.set_entry_point("retrieve")
        builder.add_conditional_edges(
            "retrieve",
            self._check_relevance,
            {"no_evidence": "no_evidence", "generate": "generate"},
        )
        builder.add_conditional_edges(
            "generate",
            self._check_answer,
            {"no_evidence": "no_evidence", "format_sources": "format_sources"},
        )
        builder.add_edge("format_sources", END)
        builder.add_edge("no_evidence", END)

        return builder.compile()

    # ── Public 인터페이스 (LCEL 버전과 동일) ──────────────────

    async def query(self, question: str) -> QueryResult:
        """
        의료 지식 RAG 쿼리 (비동기)

        Returns:
            {
                "answer": str,           # LLM 답변 (또는 no-evidence 고정 문구)
                "sources": list[dict],   # 출처 문서 목록
                "num_sources": int,      # 검색된 청크 수
            }
        """
        if self._graph is None:
            raise RuntimeError("RAG 그래프가 초기화되지 않았습니다.")

        initial_state: RAGState = {
            "question": question,
            "filtered_docs": [],
            "answer": "",
            "sources": [],
            "num_sources": 0,
            "is_no_evidence": False,
        }

        final_state = cast(RAGState, await self._graph.ainvoke(initial_state))

        return {
            "answer": final_state["answer"],
            "sources": final_state["sources"],
            "num_sources": final_state["num_sources"],
        }

    def query_sync(self, question: str) -> QueryResult:
        """동기 버전 쿼리 (테스트/스크립트용)"""
        if self._graph is None:
            raise RuntimeError("RAG 그래프가 초기화되지 않았습니다.")

        llm = self._build_llm()
        sync_graph = self._build_sync_graph(llm)

        initial_state: RAGState = {
            "question": question,
            "filtered_docs": [],
            "answer": "",
            "sources": [],
            "num_sources": 0,
            "is_no_evidence": False,
        }

        final_state = cast(RAGState, sync_graph.invoke(initial_state))

        return {
            "answer": final_state["answer"],
            "sources": final_state["sources"],
            "num_sources": final_state["num_sources"],
        }

    @property
    def is_ready(self) -> bool:
        """RAG 그래프 준비 상태 확인"""
        return self._graph is not None


# ── 로컬 테스트 ────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio

    print("Medical RAG 체인 테스트 (LangGraph 버전)")
    print("-" * 40)

    rag = MedicalRAGChain()

    test_questions = [
        "위장 폴립이란 무엇인가요?",
        "대장내시경 검사 전 준비사항은?",
        "폴립 제거 후 주의사항은?",
    ]

    async def run_tests():
        for q in test_questions:
            print(f"\n질문: {q}")
            result = await rag.query(q)
            print(f"답변: {result['answer'][:300]}...")
            print(f"출처: {result['num_sources']}개 문서")
            for src in result["sources"][:2]:
                print(f"  - {src['source_file']} p.{src['page']}")

    asyncio.run(run_tests())
