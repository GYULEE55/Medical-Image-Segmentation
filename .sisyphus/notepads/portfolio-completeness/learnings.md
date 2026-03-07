# Learnings — portfolio-completeness

## [2026-03-07] Session Start

### Project Structure
- 레포 루트: `/Users/iseung-gyu/bmad-project/i want to get a job!!/Medical-Image-Segmentation`
- 현재 구조: `project practice/` 하위에 모든 코드 존재 (루트에는 AGENTS.md, README.md, compare_csc/, example_test/, run_api.sh만 있음)
- 워킹 디렉토리: main 브랜치 단일 worktree

### Critical Issues (from plan)
- `core/` 모듈이 git에 미추적 → 클론하면 앱 실행 불가
- `.venv/`, `.pytest_cache/`가 git에 추적 중 → 첫인상 치명적
- Dockerfile에 `core/`, `vlm/`, `db/`, `eval/` COPY 누락 → Docker 빌드 실패
- `AGENTS.md`가 레포 루트에 노출 → AI 코딩 도구 사용 암시

### Key Paths
- 메인 앱: `project practice/api/app.py` (1206줄)
- 테스트: `project practice/tests/test_api.py`
- Dockerfile: `project practice/Dockerfile`
- .gitignore: `project practice/.gitignore`

## [2026-03-08] Task 5 Folder Flatten

- project practice 하위 추적 파일을 git mv로 루트로 이동해 히스토리 보존됨.
- 루트 .gitignore/README.md는 git mv -f로 교체되어 중첩 경로 안내 문구가 제거됨.
- 숨김 파일 .env는 추적 대상이 아니라 일반 mv로 루트 이동해야 import 경로(Path(__file__).parent.parent /.env)가 유지됨.
- 루트에 이미 있던 example_test와 충돌 없이 병합하려면 이미지 파일 단위로 git mv하는 것이 안전함.
- 로컬 검증은 python 대신 python3가 필요했고, python3 -m pytest tests/ -v 기준 36 passed, 1 skipped 확인됨.
- tests/test_yolo.py는 import 시 즉시 CUDA 추론하던 구조라 pytest collection 오류가 발생해, 함수형 smoke test + CPU/local sample 방식으로 안정화함.
