from pathlib import Path

from fastapi import APIRouter, HTTPException

from .. import state
from ..schemas import AskRequest, AskResponse, SourceInfo

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask_medical_knowledge(request: AskRequest):
    if state.rag_chain is None:
        raise HTTPException(
            status_code=503,
            detail="RAG 체인이 준비되지 않았습니다. 'python rag/ingest.py'를 먼저 실행하세요.",
        )

    try:
        result = await state.rag_chain.query(request.question)
        return AskResponse(
            answer=result["answer"],
            sources=[SourceInfo(**s) for s in result["sources"]],
            num_sources=result["num_sources"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 쿼리 실패: {str(e)}")


@router.get("/ask/health")
def rag_health():
    return {
        "rag_ready": state.rag_chain is not None,
        "rag_llm_provider": getattr(state.rag_chain, "provider", None),
        "vectorstore_path": str(Path(__file__).parent.parent.parent / "rag" / "vectorstore"),
        "vectorstore_exists": (
            Path(__file__).parent.parent.parent / "rag" / "vectorstore"
        ).exists(),
    }
