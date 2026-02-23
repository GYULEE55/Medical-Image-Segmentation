# 실시간 피드백 워크플로우 (Windows + MacBook)

> 목표: MacBook에서 서버 실행/테스트하면서, Windows 대화창(지금 이 채팅)에서 즉시 수정 피드백 받기

## 1) 기본 원칙

- **MacBook**: 실제 실행 환경 (ollama, uvicorn, API 테스트)
- **Windows(현재 대화)**: 코드 수정/설계/문서 정리 지시
- 코드 변경은 GitHub에 push하고, MacBook에서 pull 받아 즉시 검증

---

## 2) 가장 안정적인 방식 (권장)

### A. MacBook에서 서버 실행

```bash
cd "~/Medical-Image-Segmentation/project practice"
source venv/bin/activate
ollama serve
```

새 터미널:

```bash
cd "~/Medical-Image-Segmentation/project practice"
source venv/bin/activate
uvicorn api.app:app --reload --port 8000
```

### B. 테스트는 Mac에서 즉시 실행

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/vlm-analyze -F "file=@bus.jpg"
```

### C. 수정 요청은 Windows 채팅으로

- 실행 에러 로그를 그대로 붙여서 전달
- 나는 코드 수정 후 GitHub push
- MacBook에서 `git pull` 후 재실행

---

## 3) 반자동 루프 (수면 중에도 진행)

### MacBook에서 상시 동작 스크립트

```bash
# watch_and_test.sh
#!/bin/bash
set -e

cd "~/Medical-Image-Segmentation/project practice"
source venv/bin/activate

while true; do
  git pull --rebase || true
  python -m pytest tests/test_api.py -q || true
  sleep 300
  date
  echo "[loop] re-test completed"
done
```

```bash
chmod +x watch_and_test.sh
nohup ./watch_and_test.sh > loop.log 2>&1 &
```

- 5분마다 최신 코드 pull + 테스트 실행
- 실패 로그는 `loop.log`에 누적

---

## 4) 원격 접속으로 완전 실시간 보기

### 옵션 1: Tailscale + SSH (추천)
- MacBook에 Tailscale 설치
- 외부에서도 `ssh`로 접속해서 서버 로그 확인 가능

### 옵션 2: VSCode Remote SSH
- Windows VSCode에서 MacBook SSH 연결
- 맥 파일을 윈도우에서 직접 편집/실행처럼 다룸

---

## 5) 문제 발생 시 체크리스트

1. `No module named api`
   - 현재 경로가 `project practice`인지 확인
2. `ConnectionError to ollama`
   - `ollama serve` 실행 여부 확인
3. `/vlm-analyze` 느림
   - 첫 호출은 모델 로딩 때문에 20~60초 걸릴 수 있음
4. RAG 실패
   - `.env`에 `OPENAI_API_KEY` 확인

---

## 6) 한 줄 요약 (면접용)

> "개발 환경(Windows)과 실행 환경(MacBook)을 분리하고, Git 기반 동기화 + 자동 테스트 루프로 피드백 사이클을 짧게 유지했습니다. 로컬 VLM(ollama) 검증은 Mac에서, 코드 변경은 협업 채널에서 처리해 안정적으로 반복 개발했습니다."
