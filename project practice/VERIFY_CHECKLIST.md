# VERIFY CHECKLIST (다음 세션 시작용)

> 목적: 이전 계획(AGENTS.md)과 현재 구현 상태를 빠르게 비교/검증하기

## 0) 시작 멘트 (복붙)

아래 문장을 그대로 입력:

`AGENTS.md 기준으로 지난 계획과 현재 상태를 diff 검증해줘. 기준 커밋은 c3958b7이고, project practice/ 중심으로 계획 대비 완료/미완료를 표로 보여줘. 검증 명령 결과까지 포함해줘.`

---

## 1) Git 상태 검증

```bash
git log --oneline -5
git show --name-only --pretty=format:"%h %s" -1 c3958b7
git ls-remote medical refs/heads/main
git status --short
```

확인 포인트:
- `c3958b7` 커밋이 로컬/원격(`medical/main`)에 모두 존재
- 이번 프로젝트와 무관한 파일(예: 코테 파일)은 별도 변경으로 분리되어 있는지

---

## 2) 핵심 코드 존재 검증

```bash
grep -n "v4_vlm\|/vlm-analyze\|vlm_analyze_with_knowledge" "project practice/api/app.py"
```

확인 포인트:
- `v4_vlm` health 필드 존재
- `@app.post("/vlm-analyze")` 존재
- `def vlm_analyze_with_knowledge(...)` 존재

---

## 3) 테스트 검증

```bash
cd "project practice"
python -m pytest tests/test_api.py -q
```

기대 결과:
- `29 passed, 1 skipped` (환경에 따라 skip 수는 달라질 수 있음)

---

## 4) 실행 검증 (MacBook)

터미널 A:
```bash
ollama serve
```

터미널 B:
```bash
cd "~/Medical-Image-Segmentation/project practice"
source venv/bin/activate
uvicorn api.app:app --reload --port 8000
```

터미널 C:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/vlm-analyze -F "file=@bus.jpg"
```

확인 포인트:
- `/health`에서 `v1_yolo`, `v2_rag`, `v4_vlm` 확인
- `/vlm-analyze`에서 `vlm_analysis`, `detections`, `medical_evidence` 확인

---

## 5) 문서 동기화 검증

아래 문서가 서로 일치하는지 확인:
- `AGENTS.md`
- `project practice/README.md`
- `project practice/PROJECT_REVIEW.md`
- `project practice/SETUP_MACBOOK.md`
- `project practice/WORKFLOW_REALTIME.md`
- `project practice/JOB_PREP_PLAN.md`

검증 질문:
- V4(`vlm-analyze`) 설명이 README/PROJECT_REVIEW 모두에 반영됐는가?
- MacBook 실행 절차와 실제 실행 명령이 동일한가?
- 취준 계획 문서의 TODO가 최신 상태인가?

---

## 6) 면접 대비 30초 검증 멘트

아래 3문장으로 바로 설명 가능해야 함:

1. `V1~V4로 확장하면서 Detection → RAG → VLM+RAG 통합으로 발전시켰다.`
2. `핵심은 /vlm-analyze에서 정성(VLM) + 정량(YOLO) + 근거(RAG)를 함께 제공하는 것이다.`
3. `코드는 테스트(29 passed)와 실행 검증(health + vlm-analyze)까지 완료했다.`
